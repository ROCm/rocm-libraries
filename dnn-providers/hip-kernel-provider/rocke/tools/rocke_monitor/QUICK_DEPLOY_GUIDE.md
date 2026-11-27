<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE monitor: quick deploy guide

From an image to the monitor's first tick, in a container on this host. Before this guide comes
[QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md), which builds the image, on this machine or another. After it comes
[QUICK_USER_GUIDE.md](QUICK_USER_GUIDE.md), the day to day. [README.md](README.md) is the reference: every command,
setting and login, and, in its [glossary](README.md#glossary-and-placeholders), each term and each `<...>` placeholder
this guide uses.

The deploy path has eight steps, which [README.md, "The deploy path"](README.md#the-deploy-path) lists. Run each
command in the package folder's `docker/` folder. Each one starts with a line that names its step, such as
`== deploy step 3 of 8, confirm: check your deploy settings, show them, and confirm them`, checks what the step needs,
and ends with its outcome and a `next:` line: the next step's command or, after a failure, the same command again. A
failure says what is wrong in a `FAIL` row, and how to fix it in the row's `fix:` line. At any time, `./rkmon next`
shows where this deployment is on the path, and the command to run next.

A `next:` line names the step, as in `next: step 5, login: ./rkmon login`; after the first tick, it names the command
alone, as in `next: ./rkmon doctor`.

Below, FILE is your deploy settings file, `~/rkmon-deploy.yaml` in the examples, and IMAGE_FILE the image's file.
`--dry-run` runs a step's checks, shows what it would do, and changes nothing.

## Before you start

### Machine sizing

This host needs Linux, or WSL2, with Docker and its compose plugin (or Podman: `CONTAINER_ENGINE=podman`), and
Python 3.9 or newer for `rkmon`. Nothing is built here, so it needs no buildx. The container reaches GitHub, Claude
or your gateway, and each service a feature that is on uses: step 7 checks each one. Step 2 checks this host against
what follows, and step 3 checks the container's share of it again: the CPUs, the RAM disk and the memory. Each shows
that share in rows such as `PASS  memory: the container may use 40 of this host's 64 GiB` and
`WARN  CPUs: host.resources.cpus 2 is fewer than the reviews and builds that may run at once (4)`.

- **Disk**, where Docker keeps its images and volumes: the image is about 20 GiB with ROCm, and loading it from its
  file needs room for the file too, a third of that; the clone, its worktrees and the builds take tens more. Less
  than 30 GiB free is a FAIL, and less than 60 GiB a WARN.
- **CPUs** (`host.resources.cpus`): at least the parallel reviews and parallel builds your file starts with
  (`features.review_automation.parallel_reviews`, while review automation is on, plus `monitor.parallel_builds`), and no
  more than this host has. With fewer, each one gets less than a CPU: a WARN. More than this host has is a FAIL.
- **A RAM disk** (`host.resources.ramdisk_gib`; 0: none) makes reviews and builds faster. Until the monitor has measured
  them, it counts about 1.5 GiB for each review and 2 GiB for each build. A RAM disk smaller than the reviews and builds
  your file starts with need is a WARN. One smaller than all 5 parallel reviews and 8 parallel builds need, about 24
  GiB, is a WARN too: no more of them start at once than fit on it.
- **Memory** (`host.resources.memory_gib`): the RAM disk counts against the container's memory, so `memory_gib` must
  be at least `ramdisk_gib` plus 8, and no more than this host has: else a FAIL.

The monitor raises or lowers the parallel reviews, from 1 to 5, and the parallel builds, from 1 to 8, as the host's
load allows, starting from your file's numbers. For example, 24 CPUs, a 24 GiB RAM disk and 40 GiB of memory leave
room for all 5 parallel reviews and 8 parallel builds.

A machine that builds the image ([QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)) needs Docker with buildx,
Python 3.9 or newer, and 40 GiB free where Docker keeps its images: less is a FAIL at its step 2, and less than
80 GiB a WARN.

### The package and the image

**The package**: the same version the image was built from, as a tarball named by its version, such as
`rocke_monitor-<version>.tar.gz`, unpacked in a folder of its own. One package folder runs one deployment. The
folder, your deploy settings file and your home may sit on NFS: Docker keeps the container's volumes on this host's own
disk.

```sh
mkdir ~/rkmon && tar -xzf rocke_monitor-<version>.tar.gz -C ~/rkmon
cd ~/rkmon/rocke_monitor/docker
```

Every command below runs from that `docker/` folder as `./rkmon`. `./rkmon help` lists every command, and
`./rkmon help COMMAND` gives one command's help, with examples.

**The image**: a file the image's builder saved (`./rkmon image save`), a registry they pushed it to
(`./rkmon image push`), or an image on this host already (`./rkmon image ls`). Your deploy settings name it:
`image.registry`, `image.name` and `image.tag`. Every image carries the same tools; your settings choose which
features run.

### Pick a profile

A profile is the template step 1 writes, for one type of deployment: it decides which features start on.
`./rkmon template deploy --profile PROFILE FILE` writes the profile PROFILE's template; `full` is the default.

- `full`: reviews, the merge queue and the admin dashboard.
- `review-only`: reviews and the admin dashboard.
- `merge-queue`: the merge queue and the admin dashboard, without Claude.

Whatever the profile, every feature that posts or sends starts in prototype: it does its work, shows you what it
would post or send on the admin dashboard, and posts and sends nothing. When another deployment watches the same
repository, read "Several deployments", below, first.

### Have these ready

| # | Have ready | Where to get it | Checked at |
|---|---|---|---|
| 1 | The image: its file, or the registry it was pushed to (log in first: `docker login <registry-host>`) | the build's done line | step 4 |
| 2 | The package, the same version as the image, unpacked in a folder of its own | the image builder | step 4 |
| 3 | On a host with another deployment: a package folder, a container name and a port of your own (one package folder runs one deployment). On a repository with another deployment: whether this one runs the merge queue (one per repository), an automated reviewer name of its own, and a GitHub account of its own ("Several deployments") | `docker compose ls --all`; `ss -ltn`; the repository's other deployers | step 2 |
| 4 | The repository, `<org>/<repo>`, which must be public; the folder your project lives in there, from the repository's root (`repository.folder`, such as `projects/my-project`); and your team's scope labels | `gh label list -R <org>/<repo>`; the repository's maintainers | step 2 |
| 5 | Your GitHub account, and two fine-grained tokens that `<org>` owns, made as [Make the GitHub tokens](#make-the-github-tokens) says: a read-only one and, for live, one that may write. Ask early: `<org>` may have to approve them | GitHub, your account's developer settings | step 5 |
| 6 | With review automation: an Anthropic API key, or your gateway's address, a token for it, and the header it takes the token in. Each deployment pays for its own reviews: a key or token of its own | the Anthropic Console, or the gateway's owners | steps 5 and 7 |
| 7 | With tickets: the JIRA site's address, the project's key, the issue type when it is not "Task", and a login that may create tickets there | your JIRA admins | steps 2 and 5 |
| 8 | With alerts: a Teams Workflows webhook for the channel (the monitor sends Adaptive Cards) | the channel's owner | step 5 |
| 9 | With the team dashboard published to a web host: a web host you can SSH to, a user there, the folder it serves, that user's SSH key, and the fingerprints of the web host's own keys; for the priority box, Python 3.6 or newer on the web host, and user services that keep running after logout, for the request service `./rkmon web-host install` puts there | the web host's admin; what runs there: [README.md, "The team dashboard"](README.md#the-team-dashboard) | steps 2 and 5 |
| 10 | The share of this host to give: CPUs, memory, a RAM disk | "Machine sizing", above | step 2 |

The admin dashboard's access token is made at step 5 (`./rkmon secret admin_access_token --generate`).

#### Make the GitHub tokens

The monitor watches public repositories only, so `<org>/<repo>` must be public: [Step 5: login](#step-5-login) says
why, under its table of the GitHub logins.

Make each token on GitHub, in your account's developer settings, as a new fine-grained token, and set these fields.
The resource owner is first because it cannot be changed later: a token with another owner cannot be fixed, only made
again.

| On the page | `github_read`, the read-only token | `github_write`, the token that may write |
|---|---|---|
| Resource owner | `<org>`: only a token that `<org>` owns can have organization permissions. If `<org>` is not in the list, ask its owners to allow fine-grained tokens | `<org>` |
| Token name | a name that says which deployment uses it, such as `rkmon-my-project-read` | such as `rkmon-my-project-write` |
| Description | optional | optional |
| Expiration | the longest `<org>` allows | the longest `<org>` allows |
| Repository access | Only select repositories: `<org>/<repo>` | Only select repositories: `<org>/<repo>`, and the develop guard's issue repository when `features.merge_queue.develop_guard.issue_repository` names another one |
| Repository permissions | Actions, Commit statuses, Contents, Issues, Metadata, Pull requests: Read-only | Contents, Issues, Pull requests: Read and write; Metadata: Read-only |
| Organization permissions | Members: Read-only (who is in the code-owner teams) | none |

The develop guard's issue repository, when it is not `<org>/<repo>`, must be public too, and one that `<org>` owns.
The read-only token reads the develop issues there as every token reads a public repository, so it is not selected
for it. The token that may write reads them, opens, comments on, labels and assigns them, and makes their labels: it
needs Issues: Read and write there, and a token can be given only its resource owner's repositories.

When a token expires, its row in `./rkmon doctor` (`GitHub read login` or `GitHub write login`) fails, and its `fix:`
line names the command that sets it again. Make a new token as this section says, then set it as at step 5:
`./rkmon login github_read`, or `./rkmon login github_write --with-token`
([docs/FAQ.md, "What if a token or login expires?"](docs/FAQ.md#what-if-a-token-or-login-expires)).

`<org>` may require its owners to approve each token: ask early. Until they do, the token reads only public data, and
the monitor cannot read who is in the code-owner teams. A monitor that has read them keeps the last list, and tries
again a day later. With no list yet, when the repository's CODEOWNERS file names a team, the first start stops at
seeding the monitor's state: run `./rkmon start` again once the token is approved.

## With Claude Code

If Claude Code is on this host, it can walk you through steps 1 and 2: run `claude` in the package folder and type
`/rkmon-setup`, or `/rkmon-setup FILE` to start from a deploy file you already have. It explains each value, asks
before it looks anything up on this host, and stops before step 3, which you run yourself. It never asks for a
password, token or key: you install those at step 5, in a terminal of your own. This guide covers everything it does.

## The keys you must fill in

Step 1 writes each key the profile needs, with its comment and its default;
[README.md, "Deploy settings"](README.md#deploy-settings) explains every key. A few have no default:

<!-- BEGIN GENERATED: required-keys
     source: schema/deploy_settings.schema.json, x-profiles and x-required-when
     refresh: python3 -B tools/make_package.py --refresh-generated -->

In each profile's template, these keys have a `<...>` value for you to fill in: `repository.name`, `repository.folder`, `repository.scope_labels` and `integrations.github_cli.username`.

These keys are blank in the templates, and each needs a value when the condition beside it is true:

| Required when | Keys |
|---|---|
| remote GPU tests are on | in `features.review_automation.remote_gpu_tests`: `cluster.login_host`, `cluster.ssh_user`, `cluster.partition`, `cluster.accounts`, `cluster.gpu_types`, `node.python` |
| the develop guard is on and the merge queue is live | in `features.merge_queue.develop_guard`: `ci_red_label`, `stability_label` |
| delivery is "publish", or when serve's address is "0.0.0.0" or "::" | `features.dashboards.team.public_url` |
| delivery is "publish" | in `features.dashboards.team.publish`: `web_host`, `username`, `site_folder` |
| a type of ticket is on | `features.ticket_creation.project_key`, `integrations.atlassian_cli.jira_url` |
| api_key_sent_as is "header" or "header_and_bearer" | `model_bindings.<name>.api_key_header` |

<!-- END GENERATED: required-keys -->

Which model each pass of a review runs on, and how to bind another, such as one behind your gateway:
[README.md, "Model bindings"](README.md#model-bindings).

## Step 1: template

```sh
./rkmon template deploy --profile full ~/rkmon-deploy.yaml
```

It writes FILE from the profile's template, as a new file that only you can read (permissions 600). Fill in each
value marked `<...>`, and change any other you want: each key's comment says what the key does and which values it
takes. A feature the profile turns off shows only its `enabled: false` line (alerts and ticket creation: one for each
type): `./rkmon template section KEY` prints its section in full, every key with its comment, for when you turn it on
([QUICK_USER_GUIDE.md, "Turn on a feature"](QUICK_USER_GUIDE.md#turn-on-a-feature)).

Success: `WROTE ~/rkmon-deploy.yaml (profile full; yours only, permissions 600): fill in the 4 values marked <...>`,
then `next: step 2, check: ./rkmon check --settings ~/rkmon-deploy.yaml`.

It refuses, and writes nothing, when FILE exists or lies inside the package, a git work tree or a cloud-synced
folder: a FAIL row with its fix, then `REFUSED: nothing written` (exit 4).
[README.md, "Your private files"](README.md#your-private-files) says where your own files may go.

## Step 2: check

```sh
./rkmon check --settings ~/rkmon-deploy.yaml
```

It checks this host, this package folder and FILE, and changes nothing: Docker and its compose plugin; the host's
Python, disk, memory and CPUs ("Machine sizing"); FILE, as one of your private files; each key against the deploy
schema, with no `<...>` left; how the features' settings fit together; your team's skills folder, when FILE names
one; the dashboards' ports; the container's name; and another monitor on this host. Each WARN and FAIL row has its
fix.

Success: `PASSED: 12 checks (1 warning)`, with your counts, then
`next: step 3, confirm: ./rkmon confirm --settings ~/rkmon-deploy.yaml`. A failure ends with
`FAILED: 1 of 12 checks failed; nothing changed (fix each FAIL above)`, then the same command again: exit 3 when
Docker or its compose plugin is missing, else 4. A container name or a port that another deployment on this host
uses is a FAIL: give yours a name and a port of their own ("Several deployments").

## Step 3: confirm

```sh
./rkmon confirm --settings ~/rkmon-deploy.yaml
```

It checks FILE again and shows it: every setting, in the file's order, with what the container runs with them, the
image it names, each dashboard's address and port on this host (or the web host the team dashboard is published to),
and beside each feature's prototype or live what it does then (for the alerts in prototype: messages are written and
kept here, nothing is sent); then the checks of the whole.
A check that needs the image or
the running monitor, such as a call to each model your settings name, says which later step makes it. Then it asks
`Are these settings right?`. Your yes records FILE as this deployment's settings file, on this host: every later
command reads it, and nothing runs yet. The first start, step 6, puts the settings in place.

Success: `CONFIRMED: ~/rkmon-deploy.yaml is this deployment's settings file`, then step 4's command:
`next: step 4, image: ./rkmon image load IMAGE_FILE`, or `./rkmon image pull` when your settings name a registry.
When the image is on this host already, confirm's image row checks that it was built from this package's image
recipe, and the `next:` line skips step 4: on a first deploy, `next: step 5, login: ./rkmon login`.

A failed check ends with `FAILED: 1 of 3 checks failed; nothing confirmed (fix each FAIL above)` (exit 4); a no with
`CANCELLED: nothing confirmed` (exit 1). With no terminal to ask on, `--yes` answers yes; a failed check still
refuses. To change a setting later, edit FILE and follow
[QUICK_USER_GUIDE.md, "Change a setting"](QUICK_USER_GUIDE.md#change-a-setting).

## Step 4: image

```sh
./rkmon image load IMAGE_FILE    # the image's file
./rkmon image pull               # or: from the registry your settings name
```

Each one brings the image your settings name, then checks that it suits this package: built from this package's image
recipe (its copies of the files the image is built from, and of the few it runs from its own folder, are this
package's), and trusting this network's TLS from inside it. It changes nothing else. With the image on this host
already, there is nothing to do here: step 3's `next:` line names step 5. `./rkmon image check` runs the same checks
on the image here, whenever you want them; no `next:` line names it.

Success: `LOADED rocke-monitor:<tag>` or `PULLED rocke-monitor:<tag>`, then
`next: step 5, login: ./rkmon login`; for `./rkmon image check`, `PASSED: rocke-monitor:<tag> suits this package`.
When the file's image has another tag than your settings name, the load checks it, then names the change to make:
`next: set image.tag in ~/rkmon-deploy.yaml to "<tag>", then ./rkmon confirm`.

It fails when there is no file at IMAGE_FILE, when the image was not built from this package's image recipe, or when
this network's TLS fails inside it: a FAIL row with its fix, then `FAILED` and the same command again (exit 4). The
image recipe's fix is `fix: build an image from the new package (QUICK_BUILD_GUIDE.md), then set image.tag in your
deploy settings file to its tag`. When the file has no image of the name your settings give, there is no row: `FAILED`
names the images the file has, and its `fix:` line says to set `image.name` to that name and run `./rkmon confirm`, or
to load the file the build path saved (exit 4). It ends with exit 1 when docker could not load or pull the image.
Before step 3, `image load` loads the file and checks nothing: its `next:` line names the step still to do.

## Step 5: login

```sh
./rkmon login
```

It shows each login and secret your confirmed settings may need, in the order of the table below, each as a row:

- `PASS`: it is set.
- `FAIL`: it is missing or wrong, and the features its row names need it. Its `fix:` line is the command that sets it.
- `OFF`: nothing that is on needs it. When a feature in prototype would need it, the row says so, such as
  `features.alerts.mode is prototype: nothing goes out until it is live`.
- `WARN`: the `atlassian` login, while ticket creation is in prototype and the login is not set. Its row says what the
  login does meanwhile: tickets are prepared without it, and with it `./rkmon check` reads your JIRA project's required
  fields early. Its `fix:` line is `./rkmon login atlassian`.

On a terminal, it asks about each `FAIL` row in turn, such as `claude: set it now?`, and on a yes runs its fix there and
then: you type the value, or paste it, and nothing is shown; then it shows that row again. `./rkmon login --no-input`
asks nothing and sets nothing, on a terminal too (with a NAME, `--no-input` is a usage error). It checks the logins
in the image your confirmed settings name. Once the first start has made the container, the doctor's login rows that
warn or fail follow, under `The doctor, on the logins:`. After a new confirm, until `./rkmon apply`, one line says
instead that they come with apply, since the container still runs your earlier settings.

<!-- BEGIN GENERATED: logins
     source: rkmon_lib/secrets_registry.py
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Type | Name | Kept for | Installed by | Needed by |
|---|---|---|---|---|
| login | `github_read` | the monitor | `rkmon login github_read` | the monitor, always |
| login | `github_write` | the writer | `rkmon login github_write` | review posting; auto-merge; auto-merge guard (turning auto-merge off); auto-hold; develop issues; incidents; queue notices; drift notices; fix links |
| login | `atlassian` | the writer | `rkmon login atlassian` | follow-up tickets; out-of-scope issue tickets |
| login | `claude` | the monitor | `rkmon login claude` | Claude session |
| secret | `admin_access_token` | the monitor | `rkmon secret admin_access_token --generate` | admin dashboard |
| secret | `teams_webhook` | the writer | `rkmon secret teams_webhook` | develop break alerts; needs-you alerts |
| secret | `web_host_ssh_key` | the writer | `rkmon secret web_host_ssh_key` | team dashboard (publish) |
| secret | `gpu_ssh_key` | the monitor | `rkmon secret gpu_ssh_key` | remote GPU tests |

| Login | What to make | Repository permissions | Organization permissions |
|---|---|---|---|
| `github_read` | you, on GitHub: a fine-grained token with \<org> as its resource owner that only reads \<org>/\<repo> and \<org>'s team members | Actions, Commit statuses, Contents, Issues, Metadata, Pull requests: read | Members: read |
| `github_write` | you: gh's own sign-in, or a fine-grained token you make on GitHub with \<org> as its resource owner, for \<org>/\<repo> and the develop guard's issue repository | Contents, Issues, Pull requests: write; Metadata: read | none |

The monitor watches public repositories: no fine-grained token can read a private repository's check runs.

<!-- END GENERATED: logins -->

Each one can be set on its own, with the command its row names: `./rkmon login NAME` for a login, and
`./rkmon secret NAME` for a secret, typed, or read from a file of yours that you name after it, such as
`./rkmon secret teams_webhook ~/webhook.txt` ([README.md, "Your private files"](README.md#your-private-files) says
where such a file may go). A value is never shown, and never goes in your deploy settings file. With no terminal, a
secret whose row names `--generate` is set from a file too, such as `./rkmon secret admin_access_token FILE`:
`--generate` refuses without a terminal.

`./rkmon login github_read` takes a token alone: the read-only token, made as
[Make the GitHub tokens](#make-the-github-tokens) says, typed without echo or given on stdin. The read login has no
browser flow, and `--with-token` changes nothing for it. The token is logged in to a folder of its own first, where a
probe asks GitHub whether it can write: it comments with no body on issue or pull request 1 of `<org>/<repo>`, the
one the package's defaults name
([docs/internal_settings.md, "Package defaults"](docs/internal_settings.md#package-defaults)), and makes a branch with
no name, which GitHub refuses either way, so nothing is made. A token that can write is refused
(exit 4), with a `FAIL` row whose `fix:` line names the token to make, then the same command again. GitHub's 404 is
refused the same way, but a new token does not cure it: either `repository.name` names no repository, so set it right
in FILE, then run `./rkmon confirm` and `./rkmon apply`; or that issue is gone, so ask the package's maintainers to
name another in the repository defaults of `<org>/<repo>`
([docs/maintaining.md, "The package folder"](docs/maintaining.md#the-package-folder)). When GitHub's answer does not
say whether the token can write (a server error, a timeout, no network), it ends with `ERROR` (exit 1): run it again.
A refusal or an `ERROR` changes nothing; a token that is taken replaces the earlier read login whole.

`./rkmon login github_write --with-token` takes the token that may write, typed without echo or given on stdin.
Without `--with-token`, it runs gh's own sign-in in place of a token: that login can write to every repository your
GitHub account can write to.

`./rkmon login claude` makes one test call with the key, sent as the Claude session sends its requests, and shows its
row: `PASS` when the call is answered. If the call fails, the outcome is `ERROR`, and the key stays set. In
`./rkmon login`, this row is the one shown again after you set the key. A Claude session reads its key as it starts:
while one runs, its `SET` line names the commands that start it again, `./rkmon stop --drain`, then `./rkmon start`.

`./rkmon secret web_host_ssh_key FILE` and `./rkmon secret gpu_ssh_key FILE` set the SSH key, then show the
fingerprints of the host's own keys as the container reads them, one line each, such as `ED25519 SHA256:...`: compare
them with the ones the host's admin gave you, and accept them to record them. ssh trusts that host by the recorded
keys alone. An SSH key that is set while its host's own keys are not recorded, such as one a move in place keeps, is
a `FAIL` row whose fix is `./rkmon login`: it shows those fingerprints the same way, with no other question first.
Accepting needs a terminal: with a no, or with no terminal, the key stays set, nothing is recorded, and the row stays
a `FAIL`. When a host's own keys change, [README.md, "The hosts' own keys"](README.md#the-hosts-own-keys) says what to
do.

Success: `PASSED: every login and secret your settings need is set`, then `next: step 6, start: ./rkmon start`; with
`WARN` rows, `PASSED` counts them, such as `(1 warning)`. A login or secret still missing or wrong ends with `FAILED`
(exit 4), then the same command again; a no to each one left ends with `CANCELLED` (exit 1), then the step to do next.
Each row names the command that sets it.

## Step 6: start

```sh
./rkmon start
```

The first start puts your confirmed settings in place (`docker/.env`, the compose override and the settings volume,
which no one edits); puts this package's code in the container's code volume, so that the monitor runs it; sets up the
working volume, asking nothing: the repository's clone and its develop worktree, the tools and their virtual
environments, and the agentic files (the clone of a large repository takes some minutes); seeds the monitor's state;
and starts the container. A later start puts back what has gone missing from the working volume, and changes nothing
that is in place. A container started some other way, such as a `docker compose up` of your own, waits for its first
start and sets nothing up: `./rkmon start` stops it, then runs the first start. The local review loop's server, when
there is one, is checked again at each start until it passes both its checks, a test chat and a tool call, at the
address and with the model and key the settings give. While the monitor runs, a start does nothing
(`NOTHING TO DO: the container is running`): to start it again, `./rkmon stop --drain`, then `./rkmon start`.

Above its outcome, a start that starts the container shows what the container says as it starts, under
`what the container says as it starts:`: the entry point's lines and rk start's, with the one-time move's report after a
move in place, up to the line on the Claude side (`The Claude side: …`). What has not come within 120 seconds,
`./rkmon logs` shows.

Success: `STARTED: the monitor runs your confirmed settings`, then `next: step 7, doctor: ./rkmon doctor`, or
`next: ./rkmon doctor` once a tick has run since the start (the first may run within seconds).

It refuses, and changes nothing, before the steps it needs (settings you confirmed, the image, each login), with an
image not built from this package's image recipe, private files in the package folder that are not its own (the code
it puts in would carry them), a busy port, or a container name, compose project or volume of another deployment: each
with its fix. A start that stops on the way ends with `ERROR` (exit 1), naming what it left undone, such as "the work
volume is not set up", the lines above it saying why, and the same command again: run it once the cause is fixed, and
it goes on where the last one stopped.
Ctrl-C ends it the same way, with `INTERRUPTED` (exit 130) and where the start was, such as
`INTERRUPTED: the start stopped while setting up the work volume`, then the `next:` line.

## Step 7: doctor

```sh
./rkmon doctor
```

It shows where this deployment is on the path, and its state, as a row; then the doctor's checks, from inside the
container: what the monitor reaches (GitHub with each login, Claude or your gateway with each model the passes name,
and each service a feature that is on uses) and its own files and processes, each with its fix. `--fresh` checks
each service again, not from the few-minute cache.

Success: `PASSED: 31 checks`, with your count, then `next: step 8, first tick: ./rkmon first-tick`. Once a tick has run
since the start, `done: the monitor runs your confirmed settings` comes in its place: step 8 has passed, and
`./rkmon first-tick` shows that tick at once. A failure ends with `FAILED` and the same command again: exit 3 when a
prerequisite is missing, else 4. After a start in standby, as after a move from an existing monitor, a `WARN` row,
`standby`, is expected: it says since when and why nothing is sent, and its fix is the command that ends standby,
once `./rkmon status` and the dashboards read right.

The admin dashboard is on this host only: [README.md, "The admin dashboard"](README.md#the-admin-dashboard) says how
to reach it from your own machine, with the access token step 5 made.

## Step 8: first tick

```sh
./rkmon first-tick
```

It waits for the monitor's first tick after the start, 25 minutes at most (`--timeout-minutes N` for longer), then shows
`rk status`. While no tick has run since the start, it first asks the tick loop for one at once, and says what came of
it: `asked for a tick at once: the tick loop starts it within seconds (after the one in progress, if any)`;
`asked for a tick at once: no tick loop runs now; the next one to start takes it at once`; or, when the request cannot
be written, `no tick asked (…): the tick loop ticks at its own time`. While the monitor in the container is stopped, it
asks nothing: `REFUSED: the monitor in the container is stopped; nothing changed`, then the command that starts it.
`./rkmon apply` asks the same way while no tick has run since the start, and says what came of it in a `tick:` line,
with the same words, or `tick: no tick asked: the monitor in the container is stopped`.

Success: `PASSED: the first tick ran at …`, then
`done: the monitor runs (…). Your first day: QUICK_USER_GUIDE.md, Your first day` while the monitor runs your
settings as they are; while it is paused, its next line names the command that resumes it. After any later start,
the next lines name `./rkmon doctor`, then `./rkmon first-tick`, without the steps' numbers, then where the
deployment is. A start in standby (a move from an existing monitor, or after `./rkmon adopt`) goes on to step 7 and
step 8, and `./rkmon first-tick` then names the command that ends standby, once `./rkmon status` and the dashboards
read right. With no tick in time, it ends with
`FAILED: no tick in 25 minutes` (exit 4), and its fix names the commands that say why: `./rkmon logs --tail 50` and
`./rkmon rk status`.

## Several deployments

On one host, each deployment needs a package folder of its own (one package folder runs one deployment), a
`container.name` of its own, and ports of its own (`features.dashboards.admin.port`, and with delivery "serve",
`features.dashboards.team.serve.port`). `docker compose ls --all` lists the names in use, and `ss -ltn` the ports.
`./rkmon check` and `./rkmon start` check these on this host, and `./rkmon apply` and `./rkmon rollback` check the
ports too. A deployment's volumes go with its `container.name`, not with its folder: a new package folder given the
name of a removed deployment takes over the data that deployment kept
([QUICK_USER_GUIDE.md, "Remove a deployment"](QUICK_USER_GUIDE.md#remove-a-deployment)). Nothing else is shared, and
no deployment reads another. Two deployments whose admin dashboards you open from one machine need different
`features.dashboards.admin.port` values, on separate hosts too: the tunnel to each uses its port at both ends
([README.md, "The admin dashboard"](README.md#the-admin-dashboard)).

On one repository, deployments do not know of each other, so these rules are yours to keep:

- One deployment runs the merge queue, with the `full` or the `merge-queue` profile. Every other deployment of the
  repository uses the `review-only` profile, which never merges a pull request, puts one on hold or turns on
  auto-merge.
- Each deployment has an automated reviewer name of its own, on separate hosts too:
  `features.review_automation.automated_reviewer_name`, or its `container.name` while that is blank. Its reviews, its
  notices and its Teams messages start with it, so a reader knows which deployment sent each. A deployment follows up
  and counts only its own reviews: those whose first line gives its name.
- A team deployment reviews the pull requests that carry its scope labels (`repository.scope_labels`), with your
  team's skills when `features.review_automation.skills_folder` names them. A pull request in two deployments' scopes
  gets a review from each, as intended.
- A merge queue gates only on its own deployment's reviews. A deployment with the merge queue alone makes no reviews:
  it waits for a person's approval of each pull request, whatever reviews other deployments post.
- Each deployment pays for its own reviews: give it a Claude key or gateway token of its own
  (`./rkmon login claude`). Its spend record counts only its own reviews.
- GitHub's hourly allowance of API requests is per account. Give each deployment a GitHub account of its own. If
  deployments share one, set `integrations.github_cli.max_requests_per_hour` in each, so that together they stay
  within it.

## Moving from an existing monitor

These steps move an existing monitor into this package in place, on its own host. Your deployment keeps its
container name, its volumes and the secrets the older package kept in them, and the monitor keeps running until
step 6. Do the steps in order. Where a step runs a command of the existing package, it says so; every other command
is the new package's. Below, FOLDER is the existing package folder, and NEW the new package's folder. To move the
monitor to another machine, or to run the new deployment beside the existing one first, follow
"Beside the old monitor, on this host or another machine", below.

1. A deploy settings file. From `FOLDER/docker`, write one from the running monitor, with the existing package's own
   command: `(umask 077; ./rkmon rk settings export > ~/rkmon-deploy.yaml)` (the umask makes the file yours only).
2. The new package. Unpack it in a folder of its own ("The package and the image"), then, from `NEW/docker`, put it
   in FOLDER's place: `./rkmon package update --from NEW --folder FOLDER`. FOLDER as it was stays beside it, as
   `FOLDER.prev`: your way back (step 7). From here on, run `./rkmon` in `FOLDER/docker`. A shell that was there
   stays in the folder that became `FOLDER.prev`: `cd FOLDER/docker` again.
3. `./rkmon migrate --settings ~/rkmon-deploy.yaml`. It rewrites the file in this layout and keeps the earlier one as
   `~/rkmon-deploy.yaml.v1`. A file you wrote by hand may have secret values; migrate then refuses: move each value
   into a file only you can read, delete the `secret_values` section, and run it again. Step 5 installs them.
4. The image, then deploy steps 2 and 3. The new package runs on an image built from it
   ([QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)), not on the existing monitor's. Set `image.tag` in your file to its
   tag, and load it first, `./rkmon image load IMAGE_FILE`: the confirm reads the older package's Claude settings with
   it. Then `./rkmon check --settings ~/rkmon-deploy.yaml` and `./rkmon confirm --settings ~/rkmon-deploy.yaml`, and
   fix what the confirm's `Claude Code` rows say.
5. `./rkmon login`. It lists each login and secret still missing, such as the admin dashboard's access token, and
   installs each one. A secret the older package kept under another name counts as set: a start of the container
   copies it, once, to where this package keeps it
   ([README.md, "The one-time move's copies"](README.md#the-one-time-moves-copies)).
6. `./rkmon apply`. It drains the existing monitor, then makes the container again from this package, on its image,
   with your settings. The new monitor's start makes the one-time move: it splits the existing monitor's
   `config.json` by owner, once, and puts the monitor in standby, where it works and sends nothing, even for a feature
   that is live. The move's report, which shows no private value, is in `./rkmon logs`. Then deploy steps 7 and 8, as
   the `next:` lines say: `./rkmon doctor` and `./rkmon first-tick`.
7. Watch it for a day: `./rkmon status`, `./rkmon doctor`, and the admin dashboard with its "Not sent" list. Then run
   `./rkmon standby off`. That ends the move: `config.json` becomes `config.json.migrated`, read-only, and each older
   copy of a secret goes once that secret is set where this package keeps it; the end names each copy it keeps, with
   its fix. The older atlassian login goes, and its line says whether the atlassian login is set, with
   `./rkmon login atlassian` when it is not; the older package's header files for the local review loop and the JIRA
   MCP server go; and so does the older package's code in the work volume: from then on, there is no way back to the
   earlier package. Each feature then runs as your file says: in prototype, nothing is posted or sent until you go
   live ([QUICK_USER_GUIDE.md, "Go live"](QUICK_USER_GUIDE.md#go-live)).

   Until you end standby, you can go back to the earlier package: `./rkmon stop`, then
   `./rkmon package update --from FOLDER.prev`, which makes the two folders change places, so that FOLDER has the
   earlier package again, as it was. Then `cd FOLDER/docker` again, and run that package's own `./rkmon start`. To
   come forward again, once what failed is fixed: from `NEW/docker`,
   `./rkmon package update --from FOLDER.prev --folder FOLDER`; then `./rkmon apply` in `FOLDER/docker`.

### Beside the old monitor, on this host or another machine

The new deployment runs beside the old one, in standby, until you move over. This is also how a monitor moves to
another machine.

1. The package, in a folder of its own, and the image ("Before you start", rows 1 and 2). On the same host, the
   folder must be a new one: one package folder runs one deployment.
2. A deploy settings file: write one from the old monitor, as step 1 above says, or bring yours from the old
   machine. It is one of your private files ([README.md, "Your private files"](README.md#your-private-files) says
   where it may go): make it yours only here, `chmod 600 ~/rkmon-deploy.yaml`. When it is in an earlier layout, run
   `./rkmon migrate --settings ~/rkmon-deploy.yaml`, as step 3 above says. On the same host, give it a
   `container.name` and ports of its own ("Several deployments"). With Claude Code,
   `/rkmon-setup ~/rkmon-deploy.yaml` goes through what this machine needs.
3. Deploy steps 2 to 5: check, confirm, the image, and `./rkmon login`. Logins and secrets never travel with the
   file: install them again here.
4. The old monitor's state. On its host, copy its working folder: `docker cp CONTAINER:/work DIR`, CONTAINER being
   the old container's name; bring DIR here as you brought the file. `./rkmon adopt --from DIR --dry-run` shows what
   moves and what stays behind, with how many pull request records move and how many have our review, posted or
   kept; `./rkmon adopt --from DIR` moves it, once you say yes. Then deploy steps 6 to 8: `./rkmon start`,
   `./rkmon doctor` and `./rkmon first-tick`. The start makes the one-time move and puts the monitor in standby, as
   step 6 above says. Watch it for a day.
5. Move over. Drain and stop the old monitor, with its own package's commands, from its package folder's `docker/`
   folder: `./rkmon rk stop --drain`, then `./rkmon stop`. Here, `./rkmon stop --drain`; copy its `/work` again, and
   run `./rkmon adopt --from DIR` once more, so that this deployment starts from the old monitor's state as it
   stopped, not as the first copy left it; `./rkmon start`; then `./rkmon standby off`, which ends the move as step
   7 above says. DIR contains the monitor's state: delete each copy once the move is done.

## Removing a deployment

[QUICK_USER_GUIDE.md, "Remove a deployment"](QUICK_USER_GUIDE.md#remove-a-deployment) says how, and what stays.

## Next: QUICK_USER_GUIDE.md

Once the first tick has run, go on with
[QUICK_USER_GUIDE.md, "Your first day"](QUICK_USER_GUIDE.md#your-first-day), as step 8's done line says.
