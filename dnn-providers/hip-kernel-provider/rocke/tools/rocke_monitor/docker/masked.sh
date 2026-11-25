#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# A step of the image's build whose tool names hosts in its lines: apt, which names the package mirror in its errors
# (docker/pkgs.sh), the ROCm installer, which runs apt (docker/install_rocm.sh), and pip, which names the package index
# in its errors (docker/Dockerfile's venvs, docker/install_sdk.sh). The tool's own lines say what went wrong, so they
# are shown, masked:
# - each download location the build was given shows as its build secret's name in angle brackets (<package_mirror>):
#   its address, and its host on its own, in any case, with the user name and password before it (pip shows the user,
#   and the password as ****);
# - any other host shows as <host>, and an IP address as <address>, unless it is a public host the tools read with no
#   build secret of its own (TOOL_HOSTS, a copy of rkmon_lib/build_plan.py TOOL_HOSTS: PyPI's file host, and Ubuntu's
#   archives, key server and PPA): in an address, whose scheme, user name, password and port go with it
#   (<host>/simple/), in host='HOST' (pip), in resolve 'HOST' and resolving 'HOST' (apt, pip), and after connect to,
#   connection to and connecting to (apt, pip), located at and --trusted-host (pip). The address of a host in
#   TOOL_HOSTS shows as it is, less its user name and password;
# - on every line, apt's [IP: address port] shows as [IP: <address>], and an IP address in parentheses as (<address>).
# Such a host may be an internal one (a mirror's, an index's file host, a redirect's or a proxy's), and a build log goes
# further than the build settings file (docker/fetch.sh shows curl's failures in fixed words instead).
#
# Usage (in a build step that mounts each build secret):
#   sh masked.sh ID... -- COMMAND [ARG...]
# runs COMMAND, its standard output and standard error together on the standard output, masked, and exits with its
# exit code. Each ID is a build secret (docker/param.sh reads it): one the build was not given has no name to show, so
# its host shows as any other host does. The locations reach the filter in its environment, never on a command line.
# SECRETS_DIR moves the build secrets (the tests).
set -eu
HERE=$(dirname "$0")
TOOL_HOSTS="files.pythonhosted.org archive.ubuntu.com security.ubuntu.com ports.ubuntu.com keyserver.ubuntu.com
ppa.launchpadcontent.net"
ids=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    ids="$ids $1"
    shift
done
if [ -z "$ids" ] || [ $# -lt 2 ]; then
    echo "usage: masked.sh ID... -- COMMAND [ARG...]" >&2
    exit 2
fi
shift
given=""
for id in $ids; do
    value=$(sh "$HERE/param.sh" --optional "$id")
    [ -z "$value" ] || given="$given$id $value
"
done
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# The filter. host_as(H): how the host H shows. addresses(LINE), quoted(LINE, BEFORE) and after_words(LINE): LINE with
# each host shown so, in an address, between BEFORE and the next quote, and after one of WORDS.
{
    code=0
    "$@" 2>&1 || code=$?
    echo "$code" > "$work/code"
} | RKMON_MASKED=$given RKMON_TOOL_HOSTS=$TOOL_HOSTS awk '
function host_as(h,    l) {
    if (substr(h, 1, 1) == "<") return h
    l = tolower(h)
    if (l in public) return h
    if (l ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ || l ~ /^\[?[0-9a-f]*:[0-9a-f:.]*\]?$/) return "<address>"
    return "<host>"
}
function addresses(line,    out, url, p, auth, at, host) {
    out = ""
    while (match(line, URL)) {
        out = out substr(line, 1, RSTART - 1)
        url = substr(line, RSTART, RLENGTH)
        line = substr(line, RSTART + RLENGTH)
        while (url ~ /[.:]$/ || (url ~ /\]$/ && index(url, "[") == 0)) {
            line = substr(url, length(url)) line
            url = substr(url, 1, length(url) - 1)
        }
        p = index(url, "://")
        auth = substr(url, p + 3)
        while ((at = index(auth, "@")) > 0) auth = substr(auth, at + 1)
        host = auth
        if (substr(host, 1, 1) == "[" && index(host, "]") > 0) host = substr(host, 1, index(host, "]"))
        else sub(/:.*$/, "", host)
        if (host != "" && host_as(host) != host) out = out host_as(host)
        else out = out substr(url, 1, p + 2) auth
    }
    return out line
}
function quoted(line, before,    out, p, q, h) {
    out = ""
    while ((p = index(line, before)) > 0) {
        out = out substr(line, 1, p - 1 + length(before))
        line = substr(line, p + length(before))
        q = index(line, Q)
        if (q == 0) break
        h = substr(line, 1, q - 1)
        if (h != "") out = out host_as(h)
        line = substr(line, q)
    }
    return out line
}
function after_words(line,    out, h) {
    out = ""
    while (match(line, WORDS)) {
        out = out substr(line, 1, RSTART + RLENGTH - 1)
        line = substr(line, RSTART + RLENGTH)
        if (!match(line, /^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)/)) continue
        h = substr(line, 1, RLENGTH)
        sub(/\.+$/, "", h)
        if (h != "" && (index(h, ".") || substr(h, 1, 1) == "[" || substr(line, length(h) + 1, 1) == ":")) {
            out = out host_as(h)
            line = substr(line, length(h) + 1)
        }
    }
    return out line
}
BEGIN {
    Q = "\047"
    URL = "[A-Za-z][A-Za-z0-9+.-]*://[^/ \t\"\047<>(),;]+"
    WORDS = "([Cc]onnect(ion|ing)? to|located at|--trusted-host) "
    n = split(ENVIRON["RKMON_TOOL_HOSTS"], hosts, "[ \t\n]+")
    for (i = 1; i <= n; i++) if (hosts[i] != "") public[tolower(hosts[i])] = 1
    n = split(ENVIRON["RKMON_MASKED"], given, "\n")
    for (i = 1; i <= n; i++) {
        space = index(given[i], " ")
        if (space == 0) continue
        labels++
        label[labels] = "<" substr(given[i], 1, space - 1) ">"
        address = tolower(substr(given[i], space + 1))
        sub(/\/+$/, "", address)
        host = address
        sub(/^[a-z][a-z0-9+.-]*:\/\//, "", host)
        sub(/^[^@\/]*@/, "", host)
        if (substr(host, 1, 1) == "[") host = substr(host, 1, index(host, "]"))
        else sub(/[:\/?#].*$/, "", host)
        if (address != "") { k++; needle[k] = address; shown[k] = label[labels] }
        if (host != "" && host != address) { k++; needle[k] = host; shown[k] = label[labels] }
    }
}
{
    line = $0
    named = 0
    for (j = 1; j <= k; j++) {
        kept = ""
        while ((at = index(tolower(line), needle[j])) > 0) {
            kept = kept substr(line, 1, at - 1) shown[j]
            line = substr(line, at + length(needle[j]))
            named = 1
        }
        line = kept line
    }
    if (named) {
        for (j = 1; j <= labels; j++) gsub("[A-Za-z0-9._~%*+:-]+@" label[j], label[j], line)
    }
    line = after_words(quoted(quoted(quoted(addresses(line), "host=" Q), "resolve " Q), "resolving " Q))
    gsub(/\[IP: [^]]*\]/, "[IP: <address>]", line)
    gsub(/\([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\)/, "(<address>)", line)
    gsub(/\([0-9A-Fa-f]*:[0-9A-Fa-f]*:[0-9A-Fa-f:.]*\)/, "(<address>)", line)
    print line
    fflush()
}'
code=$(cat "$work/code" 2>/dev/null) || code=1
exit "$code"
