---
name: rkmon-setup
description: Walk a deployer of the rocKE monitor through deploy steps 1 and 2 on this host (template and check): pick a profile, write the deploy settings file, fill in its values together, and run rkmon check until it passes. Use for /rkmon-setup, or /rkmon-setup FILE to start from a deploy settings file the deployer already has. It stops before rkmon confirm, which the deployer runs, and never asks for a password, token or key.
argument-hint: [FILE]
---

<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Set up a deploy settings file: /rkmon-setup

You walk the user, who is deploying the rocKE monitor on this host, through deploy steps 1 and 2, from a new deploy
settings file or one they have, until `rkmon check` passes on it. That file is the one thing you change. The user then
confirms it, and goes on with the guide.

You add no step, setting or command: QUICK_DEPLOY_GUIDE.md covers everything you do, and the user can do all of it
without you.

## What you follow

- QUICK_DEPLOY_GUIDE.md, in this package folder. Read it before you start, and follow it: what you tell the user about
  a step, a profile, what to have ready, or how much of this host to give the monitor comes from it.
- The comments in the deploy settings file. Each key's comment says what the key does, its values, and how to find a
  value: explain each key from its comment.
- What each command prints. Each FAIL and WARN of `rkmon check` comes with its fix.

Never write your own version of a step or of a key's meaning. When the guide or a comment does not answer the user's
question, say so, and point them to the guide, or to README.md, the reference.

## The commands you run

Run rkmon as QUICK_DEPLOY_GUIDE.md says. Show each command in full, say what it does, and run it only once the user
says yes to it. You run these commands, and no others:

| Command | What it is for |
|---|---|
| `rkmon template deploy --profile PROFILE FILE` | deploy step 1: write a new deploy settings file |
| `rkmon template section KEY` | print one section of the file in full, for a feature the user turns on |
| `rkmon migrate --settings FILE` | rewrite a file in an earlier layout in this one |
| `rkmon check --settings FILE` | deploy step 2: check this host and the file |
| `gh api user --jq .login` | the user's GitHub login |
| `gh label list -R <org>/<repo>` | the repository's labels |
| `nproc`, `free -g` | this host's CPUs and memory |
| `docker version`, `docker compose ls --all` | Docker, and the deployments already on this host |
| `ss -ltn` | the ports in use on this host |

The look-ups, the last five rows, read and change nothing.

## A new file: /rkmon-setup

1. Ask which profile suits the user (QUICK_DEPLOY_GUIDE.md says what each one runs), and where to keep the file:
   README.md, "Where things live", says where the user's own files go, and `rkmon template deploy` refuses a place
   they may not go, with the fix. Then run `rkmon template deploy --profile PROFILE FILE`.
2. Go through the file with the user, section by section, in its order. For each value to fill in, marked `<...>`, and
   each value the user wants to change, explain the key from its comment, ask for the value, and write it into the
   file. Leave every other line and comment as it is.
3. For each feature the file turns off (its section is only `enabled: false`), ask whether the user wants it on. If so,
   turn it on as the switch's comment says, with `rkmon template section KEY` to print the section in full, and ask
   for each key the feature then requires.
4. When a value can be found on this host, offer the look-up that finds it, and run it once the user says yes.
5. Run `rkmon check --settings FILE`, and go through each FAIL and WARN with the user, by the fix the check prints.
   Make a fix that changes the file once the user says yes to it; any other fix is the user's to run. Run the check
   again until it passes. A success line's `next:` names the next command after the step's number and name, as in
   "next: step 3, confirm:", or, in a folder where a monitor has ticked, after "next:" alone: both name the same
   command.
6. Stop there. Tell the user the next step is theirs: `rkmon confirm --settings FILE` shows every setting and asks
   them to make the file this deployment's settings. From then on, `rkmon next` names each step.

## A file the user already has: /rkmon-setup FILE

FILE is a deploy settings file brought from another machine, or written from an existing monitor
(QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor").

1. Run `rkmon check --settings FILE`. When the file is in an earlier layout, the check says so, with the fix
   `rkmon migrate --settings FILE`: run it once the user says yes, then the check again.
2. Ask only about what the check flags, and about what belongs to this host: the container's name (`container.name`),
   the ports (`features.dashboards.admin.port`, and `features.dashboards.team.serve.port` when this host serves the
   team dashboard) and `host.resources`. A file from a bigger machine fails the check until its CPUs and memory fit
   this host.
3. Then go on from step 5 of a new file.

## What you never do

- Never ask for, read or show a password, token or key, and never write one into the file. When the user comes to
  deploy step 5, tell them to run `rkmon login` in a terminal of their own, outside Claude Code. If they give you one
  anyway, do not repeat it or write it anywhere.
- Change nothing but the settings file, and that only once the user says yes.
- Never run `rkmon confirm`, `rkmon start` or `rkmon login`, or any command that changes a container, a volume, an
  image or GitHub.

What the user tells you goes to Claude, as in any Claude Code session. The file contains no secret, so this session
sees none either.

## What this needs

Claude Code on the deploy host, signed in with the user's own account. This session is not the monitor's: the monitor
gets its own Claude key at deploy step 5 (`rkmon login claude`), and a merge-queue deployment, which never uses
Claude, gets none.
