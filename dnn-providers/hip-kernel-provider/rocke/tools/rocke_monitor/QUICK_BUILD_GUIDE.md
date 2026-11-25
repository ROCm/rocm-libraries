<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE monitor: quick build guide

From a build settings file to a tested image, saved to a file or pushed to a registry, ready for the machine that
runs the monitor. Before this guide comes [README.md](README.md): what the monitor does, and in its
[glossary](README.md#glossary-and-placeholders), each term and each `<...>` placeholder this guide uses. After it
comes [QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md), on the machine that runs the image. The two jobs can run on two
machines or on one ([One machine for both jobs](#one-machine-for-both-jobs)).

The build path has five steps, which [README.md, "The build path"](README.md#the-build-path) lists. Run each command
in the package folder's `docker/` folder. Each one starts with a line that names its step, such as
`== build step 2 of 5, check: check this machine and your build settings; change nothing`, and ends with its outcome
and a `next:` line: the next step's command or, after a failure, the same command again. A failure says what is wrong
and how to fix it. `./rkmon next --settings ~/rkmon-build.yaml` shows where that file is on the build path, and the
command to run next.

Below, FILE is your build settings file, `~/rkmon-build.yaml` in the examples; NAME:TAG is the image it names
(`image.name` and `image.tag`); and IMAGE_FILE is the file step 5 saves the image to. On steps 1, 3, 4 and 5,
`--dry-run` runs every check, shows what the step would do, and changes nothing.

## Before you start: have these ready

| # | Have ready | Where to get it | Checked at |
|---|---|---|---|
| 1 | The package, the version the deploy machine will run, unpacked in a new folder | the package's folder in a clone of the repository it is published in, or a package tarball built from it ([docs/maintaining.md, "Building the package and its tarball"](docs/maintaining.md#building-the-package-and-its-tarball)) | step 2 |
| 2 | Linux or WSL2 with Docker and buildx, and the Python and free disk that [QUICK_DEPLOY_GUIDE.md, "Machine sizing"](QUICK_DEPLOY_GUIDE.md#machine-sizing) states | | step 2 |
| 3 | The ROCm release to install (`rocm.version`): keep the template's unless your team needs another | AMD's ROCm release notes | step 2 (with your own `rocm.package_url`: step 3) |
| 4 | The atlassian tool's Linux binary for x86-64: its download address and its SHA-256 | wherever your organization keeps it; which tool it is, and how to get its SHA-256: [README.md, "Build settings"](README.md#build-settings), `atlassian_cli` | step 2 |
| 5 | Only when step 2 says this network inspects TLS: its root certificate, as a PEM file | on a machine where HTTPS works (`/usr/local/share/ca-certificates/` on Debian and Ubuntu, `/etc/pki/ca-trust/source/anchors/` on the Red Hat family), or your IT: "the root certificate of the network's TLS inspection, in PEM format" | step 2 |
| 6 | Optional: a Python package index, an apt mirror (Ubuntu bases only), a registry to push to (log in first: `docker login <registry-host>`) | your IT | step 2; the registry and its login: step 5 |

## Step 1: template

```sh
./rkmon template build ~/rkmon-build.yaml
```

It writes FILE from the build template, as a new file that only you can read. Fill in each value marked `<...>`: the
template has two, the atlassian tool's download address and its SHA-256 (row 4). Each key's comment says what it is
and what a blank value means. [README.md, "Build settings"](README.md#build-settings) lists every key, and says how to
have your editor check the file as you type.

Success: `WROTE ~/rkmon-build.yaml (yours only, permissions 600): fill in the 2 values marked <...>`, then
`next: step 2, check: ./rkmon check --settings ~/rkmon-build.yaml`.

It refuses a FILE that exists, and one where your private files may not be
([README.md, "Your private files"](README.md#your-private-files) says where they may). A refusal is a FAIL row with
its fix, a line on where private files may lie, then `REFUSED: nothing written` (exit 4) and your command again:

```text
  FAIL  private file: ~/rkmon-build.yaml exists
        fix: ./rkmon template build ~/rkmon-build-2.yaml, or move that one aside first:
             mv ~/rkmon-build.yaml ~/rkmon-build.yaml.old
What a private file is and where it may lie: README.md, "Where things live".

REFUSED: nothing written
next: ./rkmon template build ~/rkmon-build.yaml
```

## Step 2: check

```sh
./rkmon check --settings ~/rkmon-build.yaml
```

It checks this machine and FILE, and changes nothing: docker and buildx; FILE itself, as one of your private files;
each key against the build schema; the root certificates FILE names, and this network's TLS; what the build reads over
the network, such as the newest build of your ROCm release; the package folder, which may have no private file but
its own (the image would carry it); and the disk where docker keeps its images. Each problem is a FAIL row with its
fix. A WARN row is a choice for you, with its fix: a blank version takes the newest release, so two builds of FILE can
differ, and a blank SHA-256 can leave a download unchecked.

Success: `PASSED: N checks`, or `PASSED: N checks (W warnings)` when a row warns, then `The build's parameters:`, a
list of what the build will use: each value, and each download address FILE names as a build secret, such as
`build secret atlassian_cli_package_url (from atlassian_cli.package_url; never shown)`. A build secret is in no layer
of the image, no layer's history and no label. Then `next: step 3, build: ./rkmon build --settings ~/rkmon-build.yaml`.
A failure ends with `FAILED: F of N checks failed; nothing changed (fix each FAIL above)` and the same command again
(exit 4, or 3 when docker or buildx is missing).

The first check of a new FILE fails until the atlassian tool's two values are in it:
`FAIL  atlassian_cli.package_url: still the template's <URL>`, and the same for `atlassian_cli.package_sha256`, each
with its fix, which says where to get the value (row 4).

Each download address you set needs its SHA-256 too: with `rocm.package_url` set and `rocm.package_sha256` blank,
the check fails with `FAIL  rocm.package_sha256: blank, and rocm.package_url is set`. Download the file once and run
`sha256sum` on it. The same goes for `github_cli` and `claude_agent_sdk`.

With `rocm.package_url` blank, as the template leaves it, step 2 looks for your ROCm release in AMD's public folder
of installers, and fails one the folder does not list:
`FAIL  rocm.version: AMD's public folder of ROCm installers lists no build of ROCm 10.3.0`. Set `rocm.version` to a
release it lists, or `rocm.package_url` to the installer's address; step 3 then checks the release it installs.

A TLS row that fails: see [Behind a network that inspects TLS](#behind-a-network-that-inspects-tls).

## Step 3: build

```sh
./rkmon build --settings ~/rkmon-build.yaml
```

It runs step 2's checks of FILE again, then one docker build. Each value of FILE becomes a build argument, and each
download address a build secret. `rkmon build` gives docker a fixed set of flags, which `--dry-run` shows, so no build
argument or label can be added; `--no-cache` builds every layer again, none from docker's build cache.

The image keeps a record of its build at `/etc/rkmon/build.json`: each tool's version and SHA-256, and the SHA-256 of
each root certificate, with no address and no file name of yours. Its labels say which package, base and ROCm it is
([README.md, "Image labels"](README.md#image-labels)), and the deploy machine shows them; it checks that the image
was built from the image recipe of the package it deploys
([QUICK_DEPLOY_GUIDE.md, "Step 4: image"](QUICK_DEPLOY_GUIDE.md#step-4-image)). On this machine, BuildKit
keeps its own record of each build (`docker buildx history`): its build arguments and the package folder it built
from. That record stays on this machine: neither the image nor the repository carries it.

Success: a line with the image's package version, build date, base and ROCm release, then `BUILT NAME:TAG` and
`next: step 4, test: ./rkmon image test --settings ~/rkmon-build.yaml`.

When it fails:

- **A check fails**: its FAIL row, as in step 2, then
  `FAILED: F of N checks failed; nothing changed (fix each FAIL above)` and the same command again (exit 4).
- **The docker build fails**: `ERROR: the docker build exited N, so NAME:TAG was not built`, then the same command
  again (exit 1). The build's own lines above it say why:
  - `the binary prints no version`: `atlassian_cli.package_url` gives another file than the atlassian tool's Linux
    binary for the image's architecture, which the line names, and nothing is installed. Set it to that binary's
    address, and `atlassian_cli.package_sha256` to its SHA-256 (row 4).
  - A SHA-256 that differs from its `package_sha256`: the step names the SHA-256 it got, and installs nothing. Find
    out why the download changed before you change the value.
  - `install_rocm.sh: the installed ROCm reports ...`: the installer at `rocm.package_url` is of another release than
    `rocm.version`. Set `rocm.version` to the release it installs.
  - `install_rocm.sh: the installer's clang is "...", not the pinned compiler (...)`: a build of the ROCm release the
    review gates are tuned on must carry the gates' compiler, and this one does not. Set `rocm.version`, or
    `rocm.package_url`, to the ROCm build whose clang that is.
  - A download that fails says why in a few fixed words, never with its address. A timeout, a broken connection, or
    an answer of 408, 429, 500, 502, 503 or 504 is tried again, three times in all. `the server's certificate is not
    one the build trusts`: see [Behind a network that inspects TLS](#behind-a-network-that-inspects-tls).
  - The lines of apt, pip and the ROCm installer show the hosts they name masked, such as `<package_mirror>`,
    `<host>` and `[IP: <address>]`: the row "Masked name" of
    [README.md, "Glossary and placeholders"](README.md#glossary-and-placeholders) says what each stands for.

## Step 4: test

```sh
./rkmon image test --settings ~/rkmon-build.yaml
```

It runs the package's image tests on NAME:TAG, in throwaway containers; nothing leaves this host.
`./rkmon image test --help` says what each test covers, and the network each runs on.

Success: `PASSED: NAME:TAG`, then `next: step 5, ship: ./rkmon image save IMAGE_FILE --settings ~/rkmon-build.yaml`.

When it fails:

- **FILE fails its own checks** (as one of your private files, and against the build schema): its FAIL rows, as in
  step 2, then `FAILED: F of N checks failed; nothing changed (fix each FAIL above)` and the same command again
  (exit 4).
- **No image NAME:TAG here**: `REFUSED: no image NAME:TAG on this machine; nothing changed`, then
  `next: step 3, build: ./rkmon build --settings ~/rkmon-build.yaml` (exit 3). Name the build settings file the image
  was built from.
- **The tests do not start**: `ERROR: the tests of NAME:TAG did not start`, then the same command again (exit 1).
- **A test fails**: its FAIL lines, then `FAILED: NAME:TAG: its tests failed (each FAIL above)` and the same command
  again (exit 4).

## Step 5: ship

Save the image to a file, to carry to the deploy machine:

```sh
./rkmon image save ~/rocke-monitor.tar.gz --settings ~/rkmon-build.yaml
```

or push it to the registry FILE names in `image.registry`, once you are logged in to it (row 6):

```sh
./rkmon image push --settings ~/rkmon-build.yaml
```

`image save` compresses the image with gzip, or pigz when this machine has it. `image push` tags the image for the
registry, pushes it, then removes that extra tag here.

Success: `SAVED NAME:TAG to IMAGE_FILE (N GiB)` or `PUSHED NAME:TAG to the registry image.registry names`, then the
done line, one line on the terminal:

```text
done: NAME:TAG is ready (in IMAGE_FILE). On the deploy machine, follow QUICK_DEPLOY_GUIDE.md from step 1, with the
package of this version; set image.tag to "TAG" in your deploy settings, and step 4 loads IMAGE_FILE
```

After a push, it reads `(in the registry)` and `step 4 pulls it`. An image named other than `rocke-monitor` needs
`image.name` set to NAME in the deploy settings too.

When it fails:

- **FILE fails its own checks**: as in step 4 (exit 4).
- **No image NAME:TAG here**: `REFUSED: no image NAME:TAG on this machine; nothing changed`, then step 3's command
  (exit 3), as in step 4.
- **A private file in the package folder that is not the package's own**, which the image built from it may carry:
  `FAIL  the package folder: private files inside it: ...`, with its fix, then
  `FAILED: F of N checks failed; nothing changed (fix each FAIL above)` and the same command again (exit 4).
- **`image push` with no registry**: `REFUSED: your build settings name no registry (image.registry); nothing pushed`,
  with its fix: set `image.registry`, or save the image to a file (exit 4).
- **docker save, the compression or the push fails**: `ERROR: ...`, then the same command again (exit 1).

## Behind a network that inspects TLS

A network that inspects TLS opens each HTTPS connection and signs it again with its own root certificate, which the
image does not trust unless you name it. Step 2 finds out, and fails with
`FAIL  TLS: this network inspects TLS, and no root certificate is named`.

1. Get the network's root certificate as a PEM file (row 5 of "Before you start" says where).
2. Name the file in FILE's `operating_system.ca_certificates`, such as `["~/certs/my-company-root.pem"]`. Each file
   must be yours (or root's), and no one else may change it: the image trusts whatever it contains.
3. Run step 2 again: its TLS row passes once the certificates cover this network.

Name the root, the top certificate of the chain (the one that signs itself), on every base: pip, which installs the
image's Python packages, trusts a chain only up to a root. Step 2 says when the files you name have only issuing
certificates, and how to get the root. The image trusts these certificates wherever it runs. When the deploy machine's
network inspects TLS with another root, name that root too and build again: the deploy path's image step checks it.

## One machine for both jobs

To build and deploy on the same machine, run build steps 1 to 4 here, then follow
[QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md) from its step 1, with `image.tag` in your deploy settings set to the
tag you built (and `image.name` when you changed it). There is no file to carry, and no deploy step 4: with the image
here, the deploy path's step 3 names step 5 in its next line.

## Rebuilding the same package, with a tag of its own

A change to what the image carries, such as another ROCm release, a root certificate or a newer tool, needs another
build of the same package. Give each build a tag of its own, so that a deployment can move to it and back: set
`image.tag` in FILE to a tag such as `"2026.11.15-2"`, then run steps 2 to 5 again. While NAME:TAG is on this machine
already, steps 2 and 3 warn that the build would replace it:

```text
  WARN  image.tag: rocke-monitor:2026.11.15 is on this machine already; this build would replace it under the same
        tag
        fix: set image.tag in ~/rkmon-build.yaml to a tag of its own, such as "2026.11.15-2"
```

On the deploy machine, the deployment moves to the rebuilt image as
[QUICK_USER_GUIDE.md, "Move to a new version"](QUICK_USER_GUIDE.md#move-to-a-new-version) says, in its row for the
same package, rebuilt with a tag of its own.

## Next: QUICK_DEPLOY_GUIDE.md

Carry the image's file, or the name of the registry it is in, and the package of the same version to the deploy
machine, and follow [QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md) from step 1, as step 5's done line says.
