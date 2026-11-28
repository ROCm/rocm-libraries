#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Installs the atlassian CLI's Linux binary (the Dockerfile's atlassian stage: every image carries it), checked against
# its SHA-256 before anything is installed, and records what it installed (rk doctor reads /etc/rkmon/atlassian.json).
#
# Usage (while the image is built): install_atlassian.sh SHA256 [VERSION] [DEST]
#   SHA256   the binary's SHA-256 (64 hex digits; atlassian_cli.package_sha256). A binary that does not match is not
#            installed: the build stops and names the digest it got. Never change the pin to make it pass: find out why
#            first
#   VERSION  the version `atlassian version` must report (atlassian_cli.version; blank: any). A binary that reports
#            none is not installed either: it is not the atlassian tool, or not one for the image's architecture
#   DEST     the folder it goes to (default /usr/local/bin)
# The binary's download URL is the build secret atlassian_cli_package_url (build settings atlassian_cli.package_url),
# which this script reads itself (docker/param.sh) and downloads through docker/fetch.sh, so it is on no command line,
# and is never printed. Required: a blank URL stops the build (exit 5).
# RKMON_ATLASSIAN_RECORD names the record's file (default /etc/rkmon/atlassian.json); SECRETS_DIR moves the build
# secrets (the tests).
set -eu
HERE=$(dirname "$0")
want=$(printf '%s' "${1:-}" | tr 'A-F' 'a-f')
version=${2:-}
dest=${3:-/usr/local/bin}
record=${RKMON_ATLASSIAN_RECORD:-/etc/rkmon/atlassian.json}
url=$(sh "$HERE/param.sh" --optional atlassian_cli_package_url)
if [ -z "$url" ]; then
    echo "install_atlassian.sh: ATLASSIAN_CLI_PACKAGE_URL is blank (the build secret atlassian_cli_package_url), and" \
         "every image carries the atlassian CLI: set atlassian_cli.package_url in your build settings" >&2
    exit 5
fi
case "$want" in
    *[!0-9a-f]*|"") bad=1 ;;
    *) [ ${#want} -eq 64 ] && bad=0 || bad=1 ;;
esac
if [ "$bad" -eq 1 ]; then
    echo "install_atlassian.sh: ATLASSIAN_CLI_PACKAGE_SHA256 must be the binary's SHA-256 (64 hex digits): set" \
         "atlassian_cli.package_sha256 in your build settings; nothing installed" >&2
    exit 2
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if ! why=$(printf '%s\n' "$url" | sh "$HERE/fetch.sh" "$work/atlassian" 2>&1); then
    echo "install_atlassian.sh: the download from atlassian_cli.package_url failed ($why): nothing installed" >&2
    exit 4
fi
got=$(sha256sum "$work/atlassian" | cut -d' ' -f1)
if [ "$got" != "$want" ]; then
    echo "install_atlassian.sh: the downloaded binary's SHA-256 is $got, not atlassian_cli.package_sha256 ($want):" \
         "nothing installed" >&2
    exit 3
fi
mkdir -p "$dest"
install -m 755 "$work/atlassian" "$dest/atlassian"
reported=$("$dest/atlassian" version 2>/dev/null | sed -n 's/.*version v\{0,1\}\([0-9][0-9.]*\).*/\1/p' | head -1)
if [ -z "$reported" ]; then
    rm -f "$dest/atlassian"
    machine=$(uname -m)
    echo "install_atlassian.sh: the binary prints no version when run as \`atlassian version\`, so it is not the" \
         "atlassian tool, or not one for this image's architecture ($machine): nothing installed. Set" \
         "atlassian_cli.package_url to the atlassian CLI's Linux binary for $machine, and" \
         "atlassian_cli.package_sha256 to its SHA-256" >&2
    exit 3
fi
if [ -n "$version" ] && [ "$reported" != "${version#v}" ]; then
    rm -f "$dest/atlassian"
    echo "install_atlassian.sh: the binary reports version $reported, not atlassian_cli.version ($version):" \
         "nothing installed. Set atlassian_cli.package_url and atlassian_cli.package_sha256 to version $version's" \
         "binary, or atlassian_cli.version to $reported" >&2
    exit 3
fi
mkdir -p "$(dirname "$record")"
printf '{"version": "%s", "sha256": "%s"}\n' "$reported" "$got" > "$record"
echo "install_atlassian.sh: atlassian $reported installed in $dest; its SHA-256 matches ($got)"
