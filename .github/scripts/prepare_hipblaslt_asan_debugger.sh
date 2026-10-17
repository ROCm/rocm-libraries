#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail

# The GPU job runs without root. Keep packages, apt state and libraries private
# to this job; only the debugger wrapper uses the extracted libraries.
mkdir -p "$1"
tool_root=$(cd "$1" && pwd)
mkdir -p "$tool_root/bin"
existing_gdb=$(command -v gdb || true)
if [[ -n "$existing_gdb" && "${2:-}" != "--extract-packages" ]]; then
    if [[ "$existing_gdb" != "$tool_root/bin/gdb" ]]; then
        ln -sf "$existing_gdb" "$tool_root/bin/gdb"
    fi
else
    mkdir -p "$tool_root/packages" "$tool_root/root" \
        "$tool_root/lists/partial" "$tool_root/cache/archives/partial" "$tool_root/log"
    # Clear distribution hooks which can otherwise touch system cache paths.
    cat > "$tool_root/apt.conf" <<'EOF'
#clear APT::Update::Post-Invoke;
#clear APT::Update::Post-Invoke-Success;
EOF
    apt_options=(
        -c "$tool_root/apt.conf"
        -o "Dir::State::lists=$tool_root/lists"
        -o "Dir::Cache=$tool_root/cache"
        -o "Dir::Log=$tool_root/log"
        -o "APT::Sandbox::User=$(id -un)"
        -o "Acquire::Retries=2"
    )
    apt-get "${apt_options[@]}" update
    cd "$tool_root/packages"
    # Ubuntu 24.04 debugger dependencies, matching the pinned test image.
    apt-get "${apt_options[@]}" download gdb libbabeltrace1 libdebuginfod1t64 libipt2 \
        libsource-highlight4t64 libsource-highlight-common libboost-regex1.83.0 \
        libpython3.12t64 libpython3.12-stdlib libpython3.12-minimal \
        libdw1t64 libelf1t64 libmpfr6 libgmp10 libreadline8t64 libncursesw6 \
        libxxhash0 libglib2.0-0t64 libcurl3t64-gnutls libssh-4 libldap2 \
        libnghttp2-14 librtmp1 libpsl5t64 libbrotli1 libpcre2-8-0
    sha256sum ./*.deb > "$tool_root/packages.sha256"
    for package in ./*.deb; do
        dpkg-deb -x "$package" "$tool_root/root"
    done
    rm -f "$tool_root/bin/gdb"
    cat > "$tool_root/bin/gdb" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../root" && pwd)
unset LD_PRELOAD PYTHONPATH
export LD_LIBRARY_PATH="$root/usr/lib/x86_64-linux-gnu:$root/lib/x86_64-linux-gnu"
export PYTHONHOME="$root/usr"
exec "$root/usr/bin/gdb" --data-directory="$root/usr/share/gdb" "$@"
EOF
    chmod +x "$tool_root/bin/gdb"
fi
"$tool_root/bin/gdb" --version
