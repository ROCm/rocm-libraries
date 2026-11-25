<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE monitor: quick user guide

The operator's guide, from the first tick on: what to watch, how to change and grow a deployment, and what to do when
something goes wrong. `./rkmon rk guide` prints it. It follows [QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md), which
deploys the image that [QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md) builds. [README.md](README.md) is the reference,
every command and setting, with a [glossary](README.md#glossary-and-placeholders); [docs/FAQ.md](docs/FAQ.md) answers
the common questions. Each command here runs from the package folder's `docker/` folder, as `./rkmon`; FILE is your
deploy settings file.

## Your first day

The monitor starts in prototype: it reads GitHub, plans, reviews and checks, and keeps on this host what it would
have sent ([README.md, "Prototype, live and standby"](README.md#prototype-live-and-standby)). Nothing leaves this
machine until you set a feature's `mode` to "live" ("Go live"). Spend a day or two here first.

1. The admin dashboard, in every profile: reach it as [README.md, "The admin dashboard"](README.md#the-admin-dashboard),
   says. Its Needs you card lists what waits for your decision, Not sent what a feature in prototype kept instead of
   sending, and Today's log what the monitor did.
2. Reviews (profiles `full` and `review-only`). The first ticks queue every open pull request in scope
   (`repository.scope_labels`), however old, that is not a draft and has no review yet from this deployment: develop
   fixes and the pull requests you float to the top (`./rkmon rk prioritize N`, with the merge queue on) come first,
   then the others by number, lowest first. This deployment's reviews are those from your GitHub account
   (`integrations.github_cli.username`) whose first line gives its automated reviewer name
   (`features.review_automation.automated_reviewer_name`). Up to `features.review_automation.parallel_reviews` run at
   once, each once its pull request's newest push is 30 minutes old
   (`features.review_automation.schedule.push_cooldown_minutes`). Each verified review is kept as `PR_UNPOSTED_<N>.md`
   in the outputs folder ([README.md, "The outputs folder"](README.md#the-outputs-folder)), `/work` in the container,
   where `./rkmon shell` opens a shell. Going live posts none of them later, and their authors never see them: the
   follow-up tickets and the auto-merge guard's comments read only the reviews posted on GitHub. Do not post one by
   hand: the monitor records only what it posts.
3. The merge plan (profiles `full` and `merge-queue`): on the admin dashboard, each pull request with the gate it
   waits on. Without review automation, what needs judgement waits for you in Needs you.
4. Watch it work, as "Check on it" says.
5. When what you read is right, go live.

## Check on it

| Command | What it shows |
|---|---|
| `./rkmon status` | The deployment first: its container, the features on, each prototype or live, where it stands and the command to run next. Then, while the container runs, the monitor: on or off, paused or stopped, the parallel reviews and builds in effect, the review queue, the work in flight and the RAM disk |
| `./rkmon doctor` | Where the deployment is on the deploy path, then whether each setting, login, secret and server is right, each with its fix; `--fresh` skips its few-minute cache |
| `./rkmon next` | Where the deployment is on the deploy path, and the command to run next |
| `./rkmon rk features` | Each feature: on or off, prototype or live, and what it still lacks |
| `./rkmon rk roles` | Each pass's model, provider and effort |
| `./rkmon rk usage` | The tokens the reviews spent so far, and their cost while `features.review_automation.spend.enabled` is true |
| `./rkmon logs -f` | The container's log as it grows: each tick's summary, and what you should know |

The admin dashboard shows the same at a glance. With alerts on (`features.alerts`), your Teams channel gets develop
breaks and the decisions that wait for you.

## Change a setting

Edit your deploy settings file; never `docker/.env`, `docker/compose.override.yaml` or anything in the container. Then:

1. `./rkmon confirm`: it checks the file, shows each change and how it applies, and asks you to confirm. Nothing that
   runs changes yet: it ends with `CONFIRMED: 1 change waits for apply; nothing runs differently yet`, then
   `next: ./rkmon apply`.
2. `./rkmon apply`: it makes the confirmed change, with the doctor's checks before and after it. A change that restarts
   the container first lets the work in flight finish; with `--no-drain` it stops it at once, and each review starts
   again at its head after the restart, its checks from the start: keep that for a change that cannot wait. When the
   monitor does not load the change, or the doctor then finds a new failure, what ran before comes back: `ROLLED BACK`,
   with what failed and its fix. While no tick has run since the start, apply asks for one at once, and its `tick:` line
   says how that went (QUICK_DEPLOY_GUIDE.md, "Step 8: first tick"). Success:
   `APPLIED: the deployment runs your confirmed settings (1 change)`, then `next: ./rkmon doctor` when a check fails in
   the doctor after the change, even one that failed before it; else `next: ./rkmon first-tick` while no tick has run
   since the start; else `done: the monitor runs your confirmed settings`. Until deploy step 8 has passed once, the
   first two name their step: `next: step 7, doctor: ./rkmon doctor` and `next: step 8, first tick: ./rkmon first-tick`.

<!-- BEGIN GENERATED: takes-effect
     source: schema/deploy_settings.schema.json, x-applies
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| You change | It applies |
|---|---|
| `monitor.full_tick_max_interval_minutes`, `features.review_automation.schedule`, `features.review_automation.parallel_reviews`, `features.review_automation.large_prs`, `features.review_automation.spend.enabled`, `features.merge_queue.develop_guard.issue_repository`, `features.merge_queue.develop_guard.ci_red_label`, `features.merge_queue.develop_guard.ci_red_extra_labels`, `features.merge_queue.develop_guard.stability_label`, `features.weekly_report`, `integrations.github_cli.max_requests_per_hour` | at its next use; no restart |
| anything else | at a restart: rkmon apply lets the work in flight finish, then makes the container again |
| `container.name` | never on a deployment that exists: rkmon confirm refuses it |
| `updates` | when you run rkmon package update, which reads your file as it is then |

<!-- END GENERATED: takes-effect -->

A change the confirm shows with a `web host:` line reaches the web host when you run `./rkmon web-host install` after
the apply. `./rkmon rollback` puts back the code and the settings from before the last apply, with the team's skills
that went with them. It ends with `RESTORED: the deployment runs the code and settings from before your last apply`,
then `next: ./rkmon apply`: your confirmed change waits until you apply it again. While
`features.review_automation.skills_folder` names your team's skills folder, a file changed, added or removed there is a
change to confirm too. Logins and secrets are not settings: `./rkmon login NAME` and `./rkmon secret NAME` set them,
with no confirm. A change of the image itself, such as its ROCm or its tools, is a new image ("Move to a new version").

## Turn on a feature

Each feature is a section under `features` in your deploy settings file.

1. If the section in your file has only its `enabled` line (alerts and ticket creation have one for each type),
   print the whole section with `./rkmon template section KEY`, and paste it over the section in your file, from the
   section's first comment line to its last line. KEY is the section's full name, such as `features.alerts`.
2. Set `enabled` to true, and fill in each key whose comment says "Required when" that feature is on.
   `./rkmon check --settings FILE` names any key still missing, and the section to print for it, and a feature this
   one needs that is off, to turn on the same way; `./rkmon rk features` shows what each feature still lacks.
3. Make the change as "Change a setting" says. A login or secret the feature needs that is missing or wrong comes first:
   `./rkmon next` says `./rkmon login`.
4. A feature that sends starts in prototype; its `mode` key's comment says what it keeps instead of sending. Take it
   live when that reads right ("Go live").

To turn a feature off, set its `enabled` to false, as a change of setting.

## Go live

In live, a feature sends what it made in prototype: the monitor works on GitHub as `integrations.github_cli.username`
(it posts reviews and notices on other people's pull requests, requests changes to put a pull request on hold, arms
auto-merge and files issues), sends alerts to your Teams channel, publishes the team dashboard and files tickets. Take
one feature live at a time, and tell the people it reaches first. Each step is a change of setting ("Change a setting");
when a login the step needs is missing or wrong, `./rkmon next` says `./rkmon login` first. What prototype kept stays
under the admin dashboard's Not sent, and is never sent.

1. Review automation: install the GitHub write login (`./rkmon login github_write`), then set
   `features.review_automation.mode` to "live". Reviews and overlap notices are posted from the next review on.
2. The merge queue. It gates on this deployment's reviews, and no author saw a kept one: with review automation on,
   take review automation live first, or `./rkmon check` and `./rkmon confirm` warn. Rehearse an incident:
   `./rkmon rk sanity SHA --rehearsal`, SHA an older develop commit (a rehearsal sends nothing outward). The develop
   guard's two labels (`features.merge_queue.develop_guard.ci_red_label` and
   `features.merge_queue.develop_guard.stability_label`) must exist on GitHub: make a missing one there. Tell the pull
   requests' authors that the auto-merge guard starts with this step: when someone's auto-merge is on and one of the
   monitor's gates fails, it comments on the pull request and turns that auto-merge off; on your own pull requests, it
   does what `features.merge_queue.auto_merge_guard.on_your_pull_requests` says. Then set `features.merge_queue.mode` to
   "live": auto-hold, develop issues, queue and drift notices, fix links and the auto-merge guard, each as its own
   switch allows.
3. Auto-merge: when the team agrees, set `features.merge_queue.auto_merge.enabled` to true; then clear each pull
   request you want merged (`./rkmon rk clear N`).
4. Alerts: install the Teams webhook (`./rkmon secret teams_webhook FILE`), then set `features.alerts.mode` to "live".
5. The team dashboard: with `features.dashboards.team.delivery` "publish", install the SSH key for the web host
   (`./rkmon secret web_host_ssh_key FILE`) and accept the web host's own keys it shows
   ([QUICK_DEPLOY_GUIDE.md, "Step 5: login"](QUICK_DEPLOY_GUIDE.md#step-5-login)), and, for the priority box, the
   request service (`./rkmon web-host install`: [README.md, "The team dashboard"](README.md#the-team-dashboard));
   then set `features.dashboards.team.mode` to "live".
6. Tickets: install the atlassian login (`./rkmon login atlassian`), then set `features.ticket_creation.mode` to "live".

## Standby

Standby keeps the whole monitor from sending while the monitor you moved from may still send: it does all its work, even
for a feature that is live, and sends nothing. A monitor moved from an existing one starts in it, with the reason
`moved from an older install` after a move in place, or `adopted from an earlier monitor` after `./rkmon adopt`
([QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor"](QUICK_DEPLOY_GUIDE.md#moving-from-an-existing-monitor)).
What it would post, send, push or file is kept under the admin dashboard's Not sent, and never sent; turning off an
auto-merge it armed waits until standby ends; a write you type with an `rk` command is refused, naming standby
([README.md, "Prototype, live and standby"](README.md#prototype-live-and-standby)). `./rkmon status` says so; the
doctor's `standby` row and the admin dashboard's top line say since when and why. Once they read right and the monitor
you moved from has stopped, run `./rkmon standby off`: it ends standby, and with it the move or the adoption
(`RELEASED`), and the monitor sends as your settings allow. While the deployment does not run your confirmed settings,
it refuses: `REFUSED: the monitor does not run the settings you confirmed, so standby stays on; …`, then the state's fix
and `next:` line; an adoption that stopped part-way is one such state.

## Pause, stop and restart

```sh
./rkmon pause              # start no new work; the work in flight finishes. ./rkmon resume ends the pause
./rkmon stop --drain       # let the work in flight finish, then stop the container
./rkmon stop               # stop the container, once a GitHub write in flight is done
./rkmon start              # start it again; a pause stays, ./rkmon start --on ends it too
```

The container comes back by itself after a reboot or a crash, but not after `./rkmon stop`. What each level stops,
and what a restart keeps: [docs/restart.md, "The four stop levels"](docs/restart.md#the-four-stop-levels).

## Move to a new version

A newer package takes this package folder's place. `./rkmon package update --from NEW --dry-run`, NEW being the new
package's folder or archive, shows what changes, and its `next:` line names the commands your case needs. With no
`--from`, it fetches the package from the repository and branch your file's `updates` section names.

Close each `./rkmon shell` and `./rkmon rk` you have open here or in `<folder>.prev` first: the update refuses while a
command runs in either folder, naming each one and since when. While it runs, the other commands refuse with
`REFUSED: a package update runs; nothing changed`: run them again once it has ended.

| The new version | Its commands |
|---|---|
| A newer package that runs on the image you have | `./rkmon package update --from NEW`, then `./rkmon confirm` and `./rkmon apply` |
| A newer package with a new layout of the deploy settings file | `./rkmon package update --from NEW`, `./rkmon migrate --settings FILE`, then `./rkmon confirm` and `./rkmon apply` |
| A newer package that needs a new image: the dry run's `image` row is a `WARN` | `./rkmon package update --from NEW`; an image built from it ([QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)) and loaded, as below; then `./rkmon confirm` and `./rkmon apply` |
| The same package, rebuilt with a tag of its own ([QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md#rebuilding-the-same-package-with-a-tag-of-its-own)) | No package update: the image loaded, as below, then `./rkmon confirm` and `./rkmon apply` |

A new image needs a tag of its own: loaded under the tag your settings name already, it changes nothing that apply sees.
`./rkmon image load IMAGE_FILE` loads it, then names its tag as `image.tag` and `./rkmon confirm`; `./rkmon apply`
follows. From a registry: `image.tag` first, then `./rkmon confirm`, `./rkmon image pull` and `./rkmon apply`.

The package update keeps the folder it replaced, as `<folder>.prev`, and `./rkmon package update --from <folder>.prev`
puts it back; `./rkmon rollback` undoes the last apply ([README.md, "Updating"](README.md#updating)).

## Use another model for a pass

Each step of a review is a pass under `features.review_automation.passes`, with its model and effort; the section's
comments say what each pass does and which models it takes. The verification and code quality passes run on Claude. Each
other pass does nothing while its model is `default`, and the pass it helps does its work; name a model to give it that
work: `sonnet`, `haiku`, or a model binding of your own ([README.md, "Model bindings"](README.md#model-bindings), with
how to install its key; `./rkmon template section model_bindings` prints the section). Make the change as "Change a
setting" says; the next review uses it. `./rkmon rk roles` and `./rkmon doctor` check it.

## Scale, and the RAM disk

- `features.review_automation.parallel_reviews` and `monitor.parallel_builds` are where the monitor starts: it then
  moves each within its range, to how busy this host is. Lowering one stops no work: what runs finishes, and nothing
  new starts until fewer run. `./rkmon rk status` shows each, its range and why.
- The RAM disk (`host.resources.ramdisk_gib`, 0 for none) keeps each review's worktrees, builds and temporary files
  in memory, within the container's memory limit (`host.resources.memory_gib`); how large to make each:
  [QUICK_DEPLOY_GUIDE.md, "Machine sizing"](QUICK_DEPLOY_GUIDE.md#machine-sizing). A restart empties it, and the
  worktrees come back as they are needed. The monitor starts no review that does not fit. `./rkmon check` and
  `./rkmon confirm` judge both, and the CPUs, against this host; `./rkmon doctor` checks both.

## Logs, state and the outputs folder

Where the monitor keeps its state, its logs and the files it writes for you, in the container and on the host:
[README.md, "Where things live"](README.md#where-things-live). `./rkmon logs --tail 50` shows the end of the container's
log, and `./rkmon logs --since 1h` its last hour; `./rkmon shell` opens a shell in the container.

## Common problems

| Symptom | Fix |
|---|---|
| `./rkmon doctor` fails a login: not set, expired, or a read login that can write | The row's `fix:` line. Each login, and the token it needs: [README.md, "Logins and secrets"](README.md#logins-and-secrets) |
| Nothing is reviewed | `features.review_automation.enabled` is false, or `./rkmon rk orchestrate --check` names what the Claude session lacks |
| Verified reviews are not posted | Each is kept as `PR_UNPOSTED_<N>.md` ("Your first day"), and its line `UNPOSTED: …` says why: review automation is in prototype ("Go live") or the monitor in standby; review posting is off, as it runs only while `features.review_automation.enabled` is true and its `mode` is "live"; or review posting is not set up, and the line names what it needs and the fix, as `./rkmon rk features` does. A kept review is never posted later |
| `rk post` refuses a verified review (exit 4): `REFUSED: #N is in review_skip: run rk review-unskip N first`, `REFUSED: already posted on #N at head …`, or a line that ends in `: stop` | A pull request you skipped (`./rkmon rk review-skip N`): `./rkmon rk review-unskip N` takes it back. One with our review at that head already: `--again` posts another, when that is meant. A line that ends in `: stop` names what the review was verified on and has changed since, such as its head, its base, its CI, its threads or a newer comment, and what to do, as in `premise: threads changed: …: stop` |
| A feature says `OFF` or `NOT SET UP` (exit 5) | `./rkmon rk features` names what is missing ("Turn on a feature") |
| A feature's command says `OFF: ...` | Its feature is off or in prototype: nothing was done (exit 0); `./rkmon rk features` says which switch decides it |
| `./rkmon rk status` says STOPPED | After `rk stop` or `rk kill`: `./rkmon start` clears it, or `./rkmon rk start` |
| The admin dashboard does not open | `./rkmon status`, and `./rkmon doctor`, which checks its port; how to reach it: [README.md, "The admin dashboard"](README.md#the-admin-dashboard) |
| `ramdisk: NOT MOUNTED` | `host.resources.ramdisk_gib` in your deploy settings file ("Change a setting") |
| `docker ps` says unhealthy | A part keeps restarting: `/work/logs/restarts` names it, and its log says why. Or the container cannot tell whether its working volume is set up: `./rkmon logs --tail 50` shows the line that says why |
| `./rkmon next` or `./rkmon doctor` says a step is REDO | Run the command `./rkmon next` names |
| `./rkmon first-tick`: `FAILED: no tick in 25 minutes` | `./rkmon logs --tail 50` and `./rkmon rk status`: paused (`./rkmon resume`) or stopped (`./rkmon rk start`) |
| `./rkmon start` refuses an image that does not suit this package folder | The `fix:` line of each `FAIL` row above it |
| Reviews wait (`REVIEW_WAIT`) | The host is busy, or the RAM disk is full: they go on by themselves |
| A large pull request waits under Needs you | Its first review is one in-depth review in parts (`features.review_automation.large_prs.in_depth_review_over_lines`), and it starts only once you approve it: the line gives its size, about how many agents it runs and, when known, its cost. Approve it on the admin dashboard or with `./rkmon rk review-approve N`; decline it there or with `./rkmon rk review-skip N`. Its later reviews are light and need no approval |
| `./rkmon image check`: `TLS in the image` FAIL | Its `fix:` line: the image built again with this network's root certificate ([QUICK_BUILD_GUIDE.md, "Behind a network that inspects TLS"](QUICK_BUILD_GUIDE.md#behind-a-network-that-inspects-tls)), then loaded |
| `./rkmon image check`: the image was not built from this package's image recipe | Its `fix:` line: an image built from this package, its tag as `image.tag` in your deploy settings file ("Move to a new version") |
| `rk:` says the code release is damaged | `./rkmon rollback`, as that line says |

## Remove a deployment

```sh
./rkmon stop                  # once a GitHub write in flight is done
./rkmon remove --delete-data  # the container and its volumes (state, clone, secrets, logins), for good; asked first
./rkmon image ls              # this package's images on this host; docker image rm NAME:TAG removes one
```

`./rkmon remove` alone removes the container and keeps its volumes; with no container, it says `NOTHING TO DO`, and
whether data stays. `--delete-data` asks first; with no terminal to ask on, it removes nothing (exit 2) unless you add
`--yes`. Success: `REMOVED: the container and its volumes`, or before the first start,
`REMOVED: this deployment's volumes`. A volume that another deployment on this host uses stays, and the line names it.
Volumes belong to the container's name, not to a folder: what an earlier deployment of the same `container.name` left
from another package folder goes too. When docker cannot remove one, `ERROR` says what is gone and what stays, then the
command again.

What stays is yours to delete: the package folder, with the files `./rkmon` wrote in its `docker/` folder, and your
deploy settings file. The tokens the logins used stay valid until you revoke them where you made them.
