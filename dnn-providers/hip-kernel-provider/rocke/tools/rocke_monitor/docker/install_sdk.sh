#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Installs the Claude Agent SDK, which carries the Claude Code CLI, into the orchestrator's venv, and links its claude
# (docker/claude_cli.py). The SDK's wheel is downloaded and checked first, so a wheel that is not the one asked for
# installs nothing; the SDK's dependencies then come from the package index.
#
# Usage (as root, while the image is built: the Dockerfile's claude stage, which rkmon build gives each argument from
# rkmon_lib/build_plan.py):
#   install_sdk.sh VERSION SHA256 CLAUDE_CODE_VERSION VENV LINK
#   VERSION  the SDK's release, such as 0.1.0: claude_agent_sdk.version, or the newest, which rkmon build finds
#   SHA256   the wheel's SHA-256 (claude_agent_sdk.package_sha256): a mismatch installs nothing (exit 3). Leave
#            blank to skip the check.
#   CLAUDE_CODE_VERSION  the Claude Code version the wheel carries, which the installed claude must report (exit 3)
#   VENV     the venv to make (/opt/rkmon/venv-claude); LINK, the path its claude is linked at (/usr/local/bin/claude)
# The wheel is the build secret claude_agent_sdk_package_url when the build is given it (claude_agent_sdk.package_url),
# else VERSION's wheel for this machine from the package index, the build secret python_package_index
# (python_packages.package_index), which the SDK's dependencies come from either way. This script reads both itself
# (docker/param.sh) and downloads through docker/fetch.sh or pip, so neither is on a command line, and it never prints
# either: each may name an internal host. pip's own lines, which say why it failed, show through docker/masked.sh.
# A refusal names the build settings to change.
# Exits: 2, a bad argument or a build secret the build was not given; 3, a wheel or a claude that is not the one asked
# for; 4, a download that failed. SECRETS_DIR moves the build secrets (the tests).
set -eu
HERE=$(dirname "$0")
CERTS="if pip says a certificate failed to verify, name your network's root certificates in"
CERTS="$CERTS operating_system.ca_certificates"
version=${1:-}
want_sha=$(printf '%s' "${2:-}" | tr 'A-F' 'a-f')
want_cli=${3:-}
venv=${4:-}
link=${5:-}
if [ -z "$venv" ] || [ -z "$link" ]; then
    echo "usage: install_sdk.sh VERSION SHA256 CLAUDE_CODE_VERSION VENV LINK" >&2
    exit 2
fi
case "$version" in
    ""|*[!0-9A-Za-z.+_-]*) echo "install_sdk.sh: CLAUDE_AGENT_SDK_VERSION ($version) is no release such as 0.1.0:" \
                              "nothing installed" >&2; exit 2 ;;
esac
case "$want_sha" in
    "") ;;
    *[!0-9a-f]*) bad_sha=1 ;;
    *) [ ${#want_sha} -eq 64 ] && bad_sha=0 || bad_sha=1 ;;
esac
if [ "${bad_sha:-0}" -eq 1 ]; then
    echo "install_sdk.sh: CLAUDE_AGENT_SDK_PACKAGE_SHA256 must be the wheel's SHA-256 (64 hex digits): set" \
         "claude_agent_sdk.package_sha256 in your build settings; nothing installed" >&2
    exit 2
fi
if [ -z "$want_cli" ]; then
    echo "install_sdk.sh: CLAUDE_CODE_VERSION is blank: rkmon build gives the version the wheel carries; nothing" \
         "installed" >&2
    exit 2
fi
index=$(sh "$HERE/param.sh" python_package_index)
url=$(sh "$HERE/param.sh" --optional claude_agent_sdk_package_url)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/wheel"

# refuse_pip WHAT FIX CODE: no venv; pip's lines, each location in them by its build secret's name; then WHAT, which
# pip did not do, and FIX, the build settings to change, on the standard error; exit CODE.
refuse_pip() {
    rm -rf "$venv"
    sh "$HERE/masked.sh" python_package_index claude_agent_sdk_package_url -- cat "$work/pip.log" >&2 || true
    echo "install_sdk.sh: $1 (pip's lines above say why): nothing installed. $2" >&2
    exit "$3"
}

if [ -n "$url" ]; then
    # pip installs a wheel by its file's name, which must name the release; the platform tag of the name is never
    # compared with the wheel's own, so the file is named for any platform, whatever its address calls it.
    wheel="$work/wheel/claude_agent_sdk-$version-py3-none-any.whl"
    echo "install_sdk.sh: downloading the SDK's wheel (claude_agent_sdk.package_url)"
    if ! why=$(printf '%s\n' "$url" | sh "$HERE/fetch.sh" "$wheel" 2>&1); then
        echo "install_sdk.sh: the download from claude_agent_sdk.package_url failed ($why): nothing installed" >&2
        exit 4
    fi
else
    echo "install_sdk.sh: downloading claude-agent-sdk $version's wheel from the package index"
    python3 -m venv "$venv"
    if ! PIP_INDEX_URL=$index "$venv/bin/pip" download -q --no-cache-dir --no-deps --only-binary=:all: \
            -d "$work/wheel" "claude-agent-sdk==$version" > "$work/pip.log" 2>&1; then
        fix="Set claude_agent_sdk.version to a release the index has, python_packages.package_index to an index"
        fix="$fix that has it, or claude_agent_sdk.package_url to the wheel's address; $CERTS"
        refuse_pip "the package index gave no claude-agent-sdk $version wheel for this machine" "$fix" 4
    fi
    wheel=$(ls "$work/wheel"/*.whl)
fi
got_sha=$(sha256sum "$wheel" | cut -d' ' -f1)
if [ -n "$want_sha" ]; then
    if [ "$want_sha" != "$got_sha" ]; then
        rm -rf "$venv"
        echo "install_sdk.sh: the SDK's wheel's SHA-256 is $got_sha, not claude_agent_sdk.package_sha256" \
             "($want_sha): nothing installed" >&2
        exit 3
    fi
    echo "install_sdk.sh: the SDK's wheel matches claude_agent_sdk.package_sha256"
fi
[ -d "$venv" ] || python3 -m venv "$venv"
# The wheel, then its dependencies from the package index; pip refuses a wheel whose own release is not its name's.
if ! PIP_INDEX_URL=$index "$venv/bin/pip" install -q --no-cache-dir "$wheel" > "$work/pip.log" 2>&1; then
    if grep -q "inconsistent version" "$work/pip.log"; then
        fix="Set claude_agent_sdk.package_url to that release's wheel and claude_agent_sdk.package_sha256 to its"
        fix="$fix SHA-256, or claude_agent_sdk.version to the wheel's release"
        refuse_pip "the SDK's wheel is no claude-agent-sdk $version" "$fix" 3
    fi
    refuse_pip "the SDK's dependencies did not install from the package index" \
        "Set python_packages.package_index to an index that has them; $CERTS" 4
fi
"$venv/bin/python" -B "$HERE/claude_cli.py" --link "$link"
got_cli=$("$link" --version 2>/dev/null | head -1 | cut -d' ' -f1)
if [ "$got_cli" != "$want_cli" ]; then
    rm -rf "$venv"
    rm -f "$link"
    echo "install_sdk.sh: the SDK's claude reports ${got_cli:-no version}, not Claude Code $want_cli, which its wheel" \
         "names: nothing installed. The wheel is not one for this machine ($(uname -m)): set" \
         "claude_agent_sdk.package_url to claude_agent_sdk.version's wheel for it, or claude_agent_sdk.version to" \
         "another release" >&2
    exit 3
fi
echo "install_sdk.sh: claude-agent-sdk $version, with Claude Code $got_cli"
