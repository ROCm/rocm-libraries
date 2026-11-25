# check=skip=InvalidDefaultArgInFrom
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The rocKE monitor's image (QUICK_BUILD_GUIDE.md builds it, QUICK_DEPLOY_GUIDE.md runs it). Every image is full: ROCm,
# Claude Code (through the Claude Agent SDK), the atlassian CLI and the GitHub CLI; a deployment chooses its profile,
# phase and features at run time, never what the image carries. The image carries the code and the tools; everything
# of a deployment comes in at run time and is never baked in: the working folder (the volume at /work), the secrets
# (/secrets; the writer's logins and secrets in /secrets-write), the settings, the user ids (the entry point maps them at
# each start: docker/rk_ids.py) and, when used, an SSH key.
#
# Every input of the build is given on the `docker build` command line, which `rkmon build --settings FILE`
# (docker/rkmon) makes from the build settings file through rkmon_lib/build_plan.py, and `rkmon build --settings FILE
# --dry-run` prints:
# - an ordinary value is a build argument (build_plan.ARGS). None has a default here, so a build that is not given one
#   stops at the step that reads it;
# - a download location, which may name an internal host, is a build secret (build_plan.DOWNLOADS): in no layer, no
#   history and no build log. A step reads each with docker/param.sh and hands a download's address to curl on its
#   standard input (docker/fetch.sh), so it is on no command line either, and apt's and pip's own lines show the
#   package mirror and the package index by their secrets' names (docker/masked.sh);
# - the image's labels are --label flags, and its build record (/etc/rkmon/build.json) and your network's root
#   certificates are build contexts of their own (the build-record and ca-certificates stages below).
# Nothing is fetched outside this file, and nothing is changed in a container after the build.
#
# The stages: base (the OS, the certificates, the tools, Python); then, each from base and built at once by BuildKit,
# rocm (ROCm), venvs, claude, atlassian and gh. The final stage builds on the ROCm stage, because the runfile
# installer's deps=install puts system packages and alternatives there that the image must keep, and takes each other
# stage's files with COPY --from. The package's code comes last, so a code change rebuilds no tool.
#
# The base stage's image is a build argument like the others, so it has no default: a build not given one stops at the
# base stage's FROM. Docker's build check InvalidDefaultArgInFrom warns of that at every build; the parser directive on
# this file's first line skips the check (Docker reads a directive only above every comment).
ARG BASE_IMAGE

# --- your network's root certificates, when it inspects TLS: the build context of this name, which takes this stage's
# place (rkmon build gives it a private folder of copies of the files operating_system.ca_certificates names:
# build_plan.CA_CONTEXT); without one, this stage is empty. They are never in the build context of the package's code.
FROM scratch AS ca-certificates

# --- the build record: the build context of this name, which takes this stage's place (rkmon build gives it a private
# folder holding the record, build_plan.RECORD_CONTEXT); without one, the final stage finds no record and the build
# stops.
FROM scratch AS build-record

# --- base: the OS, your network's certificates, the tools, a Python of 3.12 or newer with its headers ---------------
# Your network's root certificates first (the ca-certificates stage, mounted for this one step): trusted before any
# download over TLS and by every tool at run time. bubblewrap is the review offload's sandbox; the Python headers are
# what rocKE's native build configures its pybind module with.
FROM ${BASE_IMAGE} AS base
ENV LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
COPY docker/pkgs.sh docker/param.sh docker/fetch.sh docker/masked.sh /opt/rkmon/build/
RUN --mount=type=bind,from=ca-certificates,target=/opt/rkmon/build/ca-certificates \
    --mount=type=secret,id=package_mirror \
    sh /opt/rkmon/build/pkgs.sh base /opt/rkmon/build/ca-certificates && sh /opt/rkmon/build/pkgs.sh clean
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

# --- ROCm, by AMD's runfile installer (docker/install_rocm.sh: the version, checksum, clang and hipcc checks,
# /etc/rkmon/rocm.json), from the build secret rocm_package_url ------------------------------------------------------
FROM base AS rocm
ARG ROCM_VERSION
ARG ROCM_PACKAGE_SHA256
ARG ROCM_CLANG_COMMIT
ARG ROCM_LIBRARY_TARGETS
ARG HIP_CHECK_TARGETS
COPY docker/install_rocm.sh /opt/rkmon/build/
RUN --mount=type=secret,id=rocm_package_url --mount=type=secret,id=package_mirror \
    sh /opt/rkmon/build/pkgs.sh mirror on \
    && sh /opt/rkmon/build/install_rocm.sh "${ROCM_VERSION?not given (rkmon build gives it)}" \
       "${ROCM_LIBRARY_TARGETS?not given (rkmon build gives it)}" \
       "${HIP_CHECK_TARGETS?not given (rkmon build gives it)}" \
       "${ROCM_CLANG_COMMIT?not given (rkmon build gives it)}" \
       "${ROCM_PACKAGE_SHA256?not given (rkmon build gives it)}" \
    && sh /opt/rkmon/build/pkgs.sh mirror off && sh /opt/rkmon/build/pkgs.sh clean && rm -rf /tmp/*

# --- the checks' venv and the generator suite's, from the locks (the exact versions the live checks run with) -------
FROM base AS venvs
COPY requirements/ /opt/rkmon/build/requirements/
RUN --mount=type=secret,id=python_package_index \
    PIP_INDEX_URL=$(sh /opt/rkmon/build/param.sh python_package_index) && export PIP_INDEX_URL \
    && python3 -m venv /opt/rkmon/venv \
    && sh /opt/rkmon/build/masked.sh python_package_index -- /opt/rkmon/venv/bin/pip install --no-cache-dir -q \
       -r /opt/rkmon/build/requirements/checks.lock.txt \
    && python3 -m venv /opt/rkmon/venv-gen \
    && sh /opt/rkmon/build/masked.sh python_package_index -- /opt/rkmon/venv-gen/bin/pip install --no-cache-dir -q \
       -r /opt/rkmon/build/requirements/gen.lock.txt \
    && ! /opt/rkmon/venv/bin/python -c "import torch" 2>/dev/null \
    && mkdir -p /out/opt/rkmon && mv /opt/rkmon/venv /opt/rkmon/venv-gen /out/opt/rkmon/

# --- the orchestrator's venv: the Claude Agent SDK, which carries the Claude Code CLI (docker/install_sdk.sh: the
# wheel's SHA-256, and the Claude Code version the installed claude reports) ----------------------------------------
FROM base AS claude
ARG CLAUDE_AGENT_SDK_VERSION
ARG CLAUDE_AGENT_SDK_PACKAGE_SHA256
ARG CLAUDE_CODE_VERSION
COPY docker/claude_cli.py docker/install_sdk.sh /opt/rkmon/build/
RUN --mount=type=secret,id=claude_agent_sdk_package_url --mount=type=secret,id=python_package_index \
    sh /opt/rkmon/build/install_sdk.sh "${CLAUDE_AGENT_SDK_VERSION?not given (rkmon build gives it)}" \
       "${CLAUDE_AGENT_SDK_PACKAGE_SHA256?not given (rkmon build gives it)}" \
       "${CLAUDE_CODE_VERSION?not given (rkmon build gives it)}" /opt/rkmon/venv-claude /usr/local/bin/claude \
    && mkdir -p /out/opt/rkmon /out/usr/local/bin && mv /opt/rkmon/venv-claude /out/opt/rkmon/ \
    && mv /usr/local/bin/claude /out/usr/local/bin/claude

# --- the atlassian CLI: its Linux binary, checked against its SHA-256 (docker/install_atlassian.sh) ------------------
FROM base AS atlassian
ARG ATLASSIAN_CLI_VERSION
ARG ATLASSIAN_CLI_PACKAGE_SHA256
COPY docker/install_atlassian.sh /opt/rkmon/build/
RUN --mount=type=secret,id=atlassian_cli_package_url \
    RKMON_ATLASSIAN_RECORD=/out/etc/rkmon/atlassian.json sh /opt/rkmon/build/install_atlassian.sh \
       "${ATLASSIAN_CLI_PACKAGE_SHA256?not given (rkmon build gives it)}" \
       "${ATLASSIAN_CLI_VERSION?not given (rkmon build gives it)}" /out/usr/local/bin

# --- gh: the release archive for this platform, checked against its SHA-256 (rkmon_lib/deploy.py gh_install) ---------
FROM base AS gh
ARG GITHUB_CLI_VERSION
ARG GITHUB_CLI_PACKAGE_SHA256
COPY rkmon_lib/__init__.py rkmon_lib/paths.py rkmon_lib/deploy.py /opt/rkmon/build/rkmon_lib/
RUN --mount=type=secret,id=github_cli_package_url \
    GITHUB_CLI_PACKAGE_URL=$(sh /opt/rkmon/build/param.sh github_cli_package_url) \
    && : "${GITHUB_CLI_VERSION?not given (rkmon build gives it)}" \
       "${GITHUB_CLI_PACKAGE_SHA256?not given (rkmon build gives it)}" \
    && export GITHUB_CLI_PACKAGE_URL GITHUB_CLI_VERSION GITHUB_CLI_PACKAGE_SHA256 \
    && python3 -B -c "import os, sys; sys.path.insert(0, '/opt/rkmon/build'); from rkmon_lib import deploy; \
e = os.environ.get; deploy.gh_install('/out/usr/local/bin', version=e('GITHUB_CLI_VERSION') or None, \
url=e('GITHUB_CLI_PACKAGE_URL'), sha256=e('GITHUB_CLI_PACKAGE_SHA256') or None)" \
    && /out/usr/local/bin/gh --version | head -1

# --- the image: on the ROCm stage (its system packages and alternatives stay), every other part copied in -----------
FROM rocm AS final
COPY --from=venvs /out/ /
COPY --from=claude /out/ /
COPY --from=atlassian /out/ /
COPY --from=gh /out/ /

# The users, which rkmon_lib/container_layout.py names (each start gives the monitor, its group and the writer the
# deployment's ids: docker/rk_ids.py): the monitor, which runs the session, its agents and the jobs; the writer, the
# only user that reads the writer's logins and secrets (/secrets-write, mode 700), in the monitor's group so the two
# share what they write; the 16 PR runners, one per slot, which run the code of pull requests. The runners' group
# reads the trees PR code runs from and is no group of the monitor's, so they read neither secrets volume nor the
# monitor's state; the monitor is in it, to give those trees that group. Each runner's primary group is its own, which
# no other user is in, on an id below the runners' user ids. A runner's home is in its slot (in /ramdisk), which each
# start makes (docker/rk_layout.py). Nothing of the image is every user's to write (a runner writes only its job's
# folders): /tmp and /var/tmp get the monitor's group at each start (docker/rk_layout.py), and /dev/shm and
# /dev/mqueue, mounted at each start, stay every user's.
RUN if getent passwd 1000 >/dev/null; then userdel -r "$(getent passwd 1000 | cut -d: -f1)"; fi \
    && if getent group 1000 >/dev/null; then groupdel "$(getent group 1000 | cut -d: -f1)"; fi \
    && groupadd -g 1000 rkmon && groupadd -g 1998 rkpr \
    && useradd -m -u 1000 -g rkmon -G rkpr -s /bin/bash rkmon \
    && useradd -m -u 1999 -g rkmon -s /usr/sbin/nologin rkwrite \
    && for i in $(seq 1 16); do \
         groupadd -g $((1983 - i)) rkpr$i \
         && useradd -M -d /ramdisk/slots/rkpr$i/home -u $((1999 - i)) -g rkpr$i -G rkpr -s /usr/sbin/nologin rkpr$i \
         || exit 1; \
       done \
    && mkdir -p /work /secrets /secrets-write \
    && mkdir -p -m 755 /code \
    && chown rkmon:rkmon /work /secrets && chmod 700 /secrets \
    && chown rkwrite:rkmon /secrets-write && chmod 700 /secrets-write \
    && find / -xdev -perm -0002 ! -type l -exec chmod o-w {} +

# The build record (build_plan.RECORD_PATH): what this build is made of, every file of it public. Each build's differs,
# so it comes after the tools' layers.
COPY --from=build-record --chmod=644 build.json /etc/rkmon/build.json

# The package's code, last: read-only, owned by root; rk on the PATH, and the GitHub request cap's counter
# (docker/rk_gh.py) linked as gh in /opt/rkmon/gh-counter, which the PATH below names first. Its digest goes to
# /etc/rkmon/image_code_id (a release in the code volume, /code, runs only over the image it was installed on). The
# two sudo rules, and nothing else: the session user may run the write broker as the writer and the PR-code launcher
# as any PR runner. Nothing runs as root through sudo: the start's root steps run in the entry point's root phase,
# before it drops to the user.
COPY . /opt/rocke_monitor
RUN find /opt/rocke_monitor -name __pycache__ -prune -exec rm -rf {} + \
    && chmod -R a-w,a+rX /opt/rocke_monitor \
    && printf '%s\n' 'Defaults:rkmon !requiretty' \
       'rkmon ALL=(rkwrite) NOPASSWD: /opt/rocke_monitor/docker/rk_write.py' > /etc/sudoers.d/rkmon-write \
    && printf 'rkmon ALL=(%s) NOPASSWD: /opt/rocke_monitor/docker/rk_prcode.py\n' "$(seq -s, -f 'rkpr%g' 1 16)" \
       > /etc/sudoers.d/rkmon-prcode \
    && chmod 440 /etc/sudoers.d/rkmon-write /etc/sudoers.d/rkmon-prcode \
    && visudo -cf /etc/sudoers.d/rkmon-write && visudo -cf /etc/sudoers.d/rkmon-prcode \
    && ln -s /opt/rocke_monitor/rk /usr/local/bin/rk \
    && mkdir -p /opt/rkmon/gh-counter && ln -s /opt/rocke_monitor/docker/rk_gh.py /opt/rkmon/gh-counter/gh \
    && mkdir -p /etc/rkmon \
    && (cd /opt/rocke_monitor && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) | sha256sum \
       | cut -c1-16 > /etc/rkmon/image_code_id

# The container starts as root for the entry point's root phase only: the user ids (docker/rk_ids.py), the folders'
# permissions and the slots (docker/rk_layout.py) and, at each start of `run`, the root step (docker/rk_root.py start:
# the write login, the one-time move's copies, and the code volume root's). Every command then runs as RKMON_AS (rkmon
# by default), and rk never runs as root (it drops to rkmon); root installs the code and goes back to the code kept
# (entrypoint.sh code-install and code-back, as root).
WORKDIR /work
ENV HOME=/home/rkmon RKMON_CONTAINER=1 RKMON_HOME=/work/monitor CLAUDE_CONFIG_DIR=/work/claude/config \
    DISABLE_AUTOUPDATER=1 RKMON_WRITE_BROKER=/opt/rocke_monitor/docker/rk_write.py \
    RKMON_PR_RUNNER=/opt/rocke_monitor/docker/rk_prcode.py
# Every gh call in the container goes through the GitHub request cap's counter, the first gh on the PATH each process
# inherits (rkmon_lib/gh_cap.py); the writer's rk gets the same PATH (rkmon_lib/writer_run.py BASE_ENV).
ENV PATH=/opt/rkmon/gh-counter:$PATH
VOLUME ["/work", "/secrets", "/secrets-write", "/code"]
EXPOSE 18613 18615
# Health: unhealthy while a piece the entry point keeps running crash-loops (docker/entrypoint.sh health: its restarts
# in the last 30 minutes), so `docker ps` shows it; a stopped or paused monitor, or one not set up yet, is healthy.
HEALTHCHECK --interval=5m --timeout=30s --start-period=10m --retries=2 \
    CMD ["/opt/rocke_monitor/docker/entrypoint.sh", "health"]
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/rocke_monitor/docker/entrypoint.sh"]
CMD ["run"]
