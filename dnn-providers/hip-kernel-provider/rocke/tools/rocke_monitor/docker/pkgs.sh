#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The image's one package-manager helper (docker/Dockerfile), so the recipe reads the same on every base it supports:
# apt (Ubuntu) and dnf (the RHEL 9 family: Rocky Linux, AlmaLinux, RHEL). Run as root while the image is built.
#
#   pkgs.sh family         deb or rpm (from /etc/os-release); anything else is refused (exit 2)
#   pkgs.sh trust DIR      trust the root certificates in DIR (*.crt: your network's, when it inspects TLS) for every
#                          tool, at the path the image's environment names (/etc/ssl/certs/ca-certificates.crt on both)
#   pkgs.sh base DIR       the base layer: DIR's certificates trusted before the first download over TLS, then the
#                          tools every image has (git, curl, a compiler, cmake, ninja, tini, sudo, bubblewrap, ...),
#                          git 2.44 or newer (Ubuntu 22.04 and 24.04 take it from the git-core archive), and a Python
#                          of 3.12 or newer with its headers, as `python3` and `python` on the PATH (the checks' locks
#                          need 3.12: Ubuntu 22.04 takes it from the deadsnakes archive, the RHEL 9 family from its
#                          AppStream)
#   pkgs.sh install NAME...  more packages, by the apt names below (each mapped to its dnf name)
#   pkgs.sh mirror on|off  apt's archives at the address the build secret package_mirror gives (build settings
#                          operating_system.package_mirror), for the steps between on and off, and apt quiet
#                          meanwhile (it prints each archive's address as it reads it); off puts the base's own
#                          sources and apt's settings back, so no layer keeps the address. Without the secret both
#                          do nothing. The mirror is for the Ubuntu bases only: on a base of the RHEL family, which
#                          keeps its own mirror list, `on` with the secret stops the build (exit 2), as rkmon check
#                          refuses the two together. `base` turns it on and off itself
#   pkgs.sh clean          the package manager's caches, out of the layer
# PKGS_APT_DIR moves /etc/apt and SECRETS_DIR /run/secrets (the tests).
set -eu
HERE=$(dirname "$0")

family() {
    id=""
    like=""
    osr=${PKGS_OS_RELEASE:-/etc/os-release}  # another file: the tests'
    if [ -r "$osr" ]; then
        # shellcheck disable=SC1090
        . "$osr"
        id=${ID:-}
        like=${ID_LIKE:-}
    fi
    case " $id $like " in
        *" debian "*|*" ubuntu "*) echo deb ;;
        *" rhel "*|*" fedora "*|*" centos "*|*" rocky "*|*" almalinux "*) echo rpm ;;
        *) echo "pkgs.sh: an unsupported base ($id): use Ubuntu or the RHEL 9 family" >&2; return 2 ;;
    esac
}

# An apt name, as dnf calls it ("" when the RHEL family has it built in).
rpm_name() {
    case "$1" in
        openssh-client) echo openssh-clients ;;
        python3-dev) echo python3-devel ;;
        python3-venv|python-is-python3) echo "" ;;
        procps) echo procps-ng ;;
        iproute2) echo iproute ;;
        xz-utils) echo xz ;;
        g++) echo gcc-c++ ;;
        *) echo "$1" ;;
    esac
}

install() {
    case $F in
        deb)
            sh "$HERE/masked.sh" package_mirror -- apt-get update -q
            sh "$HERE/masked.sh" package_mirror -- apt-get install -y --no-install-recommends "$@"
            ;;
        rpm)
            names=""
            for n in "$@"; do
                r=$(rpm_name "$n")
                [ -n "$r" ] && names="$names $r"
            done
            # shellcheck disable=SC2086
            [ -n "$names" ] && dnf install -y --setopt=install_weak_deps=False --allowerasing $names
            ;;
    esac
}

# The apt archives at the build's mirror (on), and back (off). The address is never printed: apt, which names each
# archive as it reads it (its Get, Hit and Ign lines), runs at its quiet level 2 meanwhile (QUIET, in apt's settings
# folder), which prints errors only, and install shows the mirror in those by its secret's name (docker/masked.sh).
QUIET=00rkmon-mirror-quiet
mirror() {
    apt=${PKGS_APT_DIR:-/etc/apt}
    case "${1:-}" in
        on)
            m=$(sh "$HERE/param.sh" --optional package_mirror)
            [ -n "$m" ] || return 0
            if [ "$F" != deb ]; then
                echo "pkgs.sh: the build was given a package mirror (operating_system.package_mirror), which is" \
                     "for the Ubuntu base images only: delete it from your build settings, or build on an" \
                     "Ubuntu base; nothing installed" >&2
                return 2
            fi
            m=$(printf '%s' "${m%/}" | sed 's/[&#\\]/\\&/g')
            for f in "$apt/sources.list" "$apt"/sources.list.d/*.list "$apt"/sources.list.d/*.sources; do
                [ -f "$f" ] || continue
                grep -qE 'https?://(archive|security|ports)\.ubuntu\.com' "$f" || continue
                [ -f "$f.rkmon-orig" ] || cp -p "$f" "$f.rkmon-orig"
                sed -i -E "s#https?://(archive|security)\.ubuntu\.com/ubuntu/?#$m/#g; \
s#https?://ports\.ubuntu\.com/ubuntu-ports/?#$m/#g" "$f"
            done
            mkdir -p "$apt/apt.conf.d"
            echo 'quiet "2";' > "$apt/apt.conf.d/$QUIET"
            echo "pkgs.sh: apt reads the build's mirror (operating_system.package_mirror), quietly, until" \
                 "\`pkgs.sh mirror off\`"
            ;;
        off)
            for o in "$apt/sources.list.rkmon-orig" "$apt"/sources.list.d/*.rkmon-orig; do
                [ -f "$o" ] && mv -f "$o" "${o%.rkmon-orig}"
            done
            rm -f "$apt/apt.conf.d/$QUIET"
            return 0
            ;;
        *) echo "usage: pkgs.sh mirror on|off" >&2; return 2 ;;
    esac
}

trust() {
    dir=${1:-}
    case $F in
        deb)
            if [ -n "$dir" ] && ls "$dir"/*.crt >/dev/null 2>&1; then
                mkdir -p /usr/local/share/ca-certificates/rkmon
                cp "$dir"/*.crt /usr/local/share/ca-certificates/rkmon/
            fi
            command -v update-ca-certificates >/dev/null && update-ca-certificates >/dev/null
            ;;
        rpm)
            if [ -n "$dir" ] && ls "$dir"/*.crt >/dev/null 2>&1; then
                cp "$dir"/*.crt /etc/pki/ca-trust/source/anchors/
            fi
            update-ca-trust
            # The image's environment names the Debian path (SSL_CERT_FILE and the rest): the same bundle there.
            mkdir -p /etc/ssl/certs
            [ -e /etc/ssl/certs/ca-certificates.crt ] \
                || ln -s /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt
            ;;
    esac
}

# An Ubuntu archive on Launchpad (OWNER/ppa), signed by the key with fingerprint KEY, read from Ubuntu's key server
# through docker/fetch.sh (tried again on a passing failure).
ppa() {
    owner=$1
    key=$2
    # shellcheck disable=SC1090
    . "${PKGS_OS_RELEASE:-/etc/os-release}"
    install gpg
    mkdir -p /etc/apt/keyrings
    if ! why=$(printf '%s\n' "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x$key" \
            | sh "$HERE/fetch.sh" "/tmp/$owner.asc" 2>&1); then
        echo "pkgs.sh: the signing key of the $owner archive did not download ($why)" >&2
        return 4
    fi
    gpg --dearmor -o "/etc/apt/keyrings/$owner.gpg" "/tmp/$owner.asc"
    rm -f "/tmp/$owner.asc"
    echo "deb [signed-by=/etc/apt/keyrings/$owner.gpg] https://ppa.launchpadcontent.net/$owner/ppa/ubuntu ${VERSION_CODENAME} main" \
        >"/etc/apt/sources.list.d/$owner.list"
}

# Python 3.12 or newer: the checks' locks need it.
python_new_enough() {
    command -v python3 >/dev/null && python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null
}

# git 2.44 or newer: the monitor reads its partial clone with GIT_NO_LAZY_FETCH, which older git ignores.
git_new_enough() {
    v=$(git --version 2>/dev/null | sed -n 's/^git version \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
    [ -n "$v" ] && set -- $v && { [ "$1" -gt 2 ] || { [ "$1" -eq 2 ] && [ "$2" -ge 44 ]; }; }
}

git_tool() {
    if [ "$F" = deb ] && ! git_new_enough; then
        # Ubuntu 22.04 and 24.04: git from the git-core archive (Ubuntu's own git maintainers).
        ppa git-core E1DD270288B4E6030699E45FA1715D88E1DF1F24
        install git
    fi
    git_new_enough || { echo "pkgs.sh: git $(git --version) is older than 2.44" >&2; return 2; }
}

python() {
    case $F in
        deb)
            install python3 python3-venv python3-pip python3-dev python-is-python3
            if ! python_new_enough; then
                # Ubuntu 22.04: Python 3.12 from the deadsnakes archive.
                ppa deadsnakes F23C5A6CF475977595C89F51BA6932366A755776
                install python3.12 python3.12-dev python3.12-venv
                link_python /usr/bin/python3.12
            fi
            ;;
        rpm)
            dnf install -y --setopt=install_weak_deps=False python3.12 python3.12-devel python3.12-pip
            link_python /usr/bin/python3.12
            ;;
    esac
    python3 -c "import os, sys, sysconfig; h = os.path.join(sysconfig.get_path('include'), 'Python.h'); \
assert sys.version_info >= (3, 12), sys.version; assert os.path.isfile(h), h"
}

# python3 and python on the PATH (/usr/local/bin, ahead of /usr/bin) are PY; the system's own python3 stays as it is
# for the package manager.
link_python() {
    ln -sf "$1" /usr/local/bin/python3
    ln -sf "$1" /usr/local/bin/python
    hash -r  # the shell found the older python3 before: look again
}

base() {
    mirror on
    case $F in
        deb)
            # apt reaches an https mirror before ca-certificates is installed: your certificates are its first bundle
            # (update-ca-certificates writes the whole one below).
            if [ -n "${1:-}" ] && ls "$1"/*.crt >/dev/null 2>&1 && [ ! -s /etc/ssl/certs/ca-certificates.crt ]; then
                mkdir -p /etc/ssl/certs
                cat "$1"/*.crt > /etc/ssl/certs/ca-certificates.crt
            fi
            install ca-certificates
            trust "${1:-}"
            install curl git openssh-client tini procps util-linux iproute2 rsync jq patch diffutils file \
                xz-utils less cmake ninja-build g++ make bubblewrap sudo
            ;;
        rpm)
            trust "${1:-}"
            dnf install -y --setopt=install_weak_deps=False dnf-plugins-core epel-release
            dnf config-manager --set-enabled crb
            install curl git openssh-client tini procps util-linux iproute2 rsync jq patch diffutils file \
                xz-utils less cmake ninja-build g++ make bubblewrap sudo findutils which tar gzip
            ;;
    esac
    git_tool
    python
    mirror off
}

clean() {
    case $F in
        deb) rm -rf /var/lib/apt/lists/* ;;
        rpm) dnf clean all >/dev/null; rm -rf /var/cache/dnf ;;
    esac
}

cmd=${1:-}
[ $# -gt 0 ] && shift
case "$cmd" in
    family|trust|base|install|mirror|clean) F=$(family) || exit 2 ;;
esac
case "$cmd" in
    family) echo "$F" ;;
    trust) trust "$@" ;;
    base) base "$@" ;;
    install) install "$@" ;;
    mirror) mirror "$@" ;;
    clean) clean ;;
    *) echo "usage: pkgs.sh family | trust DIR | base DIR | install NAME... | mirror on|off | clean" >&2; exit 2 ;;
esac
