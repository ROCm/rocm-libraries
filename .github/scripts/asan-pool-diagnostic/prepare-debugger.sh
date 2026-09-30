#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail

# Extract packages into this job's directory without changing the image or host.
tool_root="$PWD/diagnostic-tools"
mkdir -p "$tool_root/packages" "$tool_root/root" diagnostics
if command -v gdb >/dev/null && [[ "${1:-}" != "--extract-packages" ]]; then
    command -v gdb > diagnostics/debugger-path.txt
else
    # The image's package lists can be stale. Refresh a private copy; apt never
    # installs packages or writes to the system package database here.
    mkdir -p "$tool_root/lists/partial" "$tool_root/cache/archives/partial"
    apt_options=(
        -o "Dir::State::lists=$tool_root/lists"
        -o "Dir::Cache=$tool_root/cache"
        -o "APT::Sandbox::User=$(id -un)"
        -o "APT::Update::Post-Invoke::="
        -o "APT::Update::Post-Invoke-Success::="
    )
    apt-get "${apt_options[@]}" update
    cd "$tool_root/packages"
    apt-get "${apt_options[@]}" download gdb libbabeltrace1 libdebuginfod1t64 libipt2 \
        libsource-highlight4t64 libsource-highlight-common libboost-regex1.83.0 \
        libpython3.12t64 libpython3.12-stdlib libpython3.12-minimal \
        libdw1t64 libelf1t64 libmpfr6 libgmp10 libreadline8t64 libncursesw6 \
        libxxhash0 libglib2.0-0t64 libcurl3t64-gnutls libssh-4 libldap2 \
        libnghttp2-14 librtmp1 libpsl5t64 libbrotli1 libpcre2-8-0
    for package in ./*.deb; do
        dpkg-deb -x "$package" "$tool_root/root"
    done
    sha256sum ./*.deb > "$GITHUB_WORKSPACE/diagnostics/debugger-packages.txt"
    echo "$tool_root/root/usr/bin/gdb" > "$GITHUB_WORKSPACE/diagnostics/debugger-path.txt"
fi
