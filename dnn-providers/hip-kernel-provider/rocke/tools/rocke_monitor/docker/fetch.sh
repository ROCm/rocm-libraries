#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# One download of the image's build (docker/install_rocm.sh, docker/install_atlassian.sh, docker/install_sdk.sh and
# docker/pkgs.sh), whose address may name a private host. The address comes on the standard input and goes to curl
# the same way (curl --config -), so it is on no command line: every user of the build host can list those. curl's
# own messages name the host, so they are not shown: a download that fails says why in a fixed message for curl's
# exit code.
#
# A failure worth another attempt is tried again, as the build's downloads in Python are (rkmon_lib/deploy.py:
# RETRIED_STATUS and its comment, FETCH_ATTEMPTS, FETCH_PAUSE_SECONDS): `attempts` in all at most, `pause` seconds
# apart, for an HTTP answer of 408, 429, 500, 502, 503 or 504, a timeout, or a connection that broke (curl's exit
# codes 18, 28, 52, 55 and 56). Any other failure stands: a refusal, a name that does not resolve, a refused
# connection, a certificate the build does not trust.
#
# Usage:
#   printf '%s\n' "$url" | sh fetch.sh FILE
# downloads the address into FILE. Exit 0 when it downloaded; else curl's exit code (2 when no address came), and one
# line on the standard error that says why and names no host. FETCH_PAUSE_SECONDS moves the pause (the tests).
set -u
file=${1:?usage: printf '%s\n' URL | sh fetch.sh FILE}
url=""
IFS= read -r url || true
if [ -z "$url" ]; then
    echo "no address came on the standard input" >&2
    exit 2
fi
# curl's config syntax: the value in double quotes, each backslash and double quote in it escaped.
quoted=$(printf '%s' "$url" | sed 's/[\\"]/\\&/g')
attempts=3
pause=${FETCH_PAUSE_SECONDS:-5}
attempt=1
while :; do
    code=0
    status=$(printf 'url = "%s"\n' "$quoted" | curl -fsSL --config - -o "$file" -w '%{http_code}' 2>/dev/null) \
        || code=$?
    [ "$code" -eq 0 ] && exit 0
    case $code in
        18|28|52|55|56) again=yes ;;
        22) case $status in 408|429|500|502|503|504) again=yes ;; *) again=no ;; esac ;;
        *) again=no ;;
    esac
    [ "$again" = yes ] && [ "$attempt" -lt "$attempts" ] || break
    attempt=$((attempt + 1))
    sleep "$pause"
done
case $code in
    3) why="the address is no URL curl can read" ;;
    5) why="the proxy's name does not resolve" ;;
    6) why="the host's name does not resolve" ;;
    7) why="no connection to the host: it refused, or cannot be reached" ;;
    18) why="the download ended before the whole file came" ;;
    22) why="the server answered HTTP ${status:-400 or above}" ;;
    23) why="the file could not be written here" ;;
    28) why="it timed out" ;;
    35) why="the TLS handshake failed" ;;
    47) why="too many redirects" ;;
    52) why="the server answered nothing" ;;
    55|56) why="the connection broke during the download" ;;
    60) why="the server's certificate is not one the build trusts (a network that inspects TLS: name its root"
        why="$why certificates in operating_system.ca_certificates)" ;;
    *) why="curl could not download it" ;;
esac
echo "$why, curl exit $code" >&2
exit "$code"
