#!/usr/bin/env bash
# Start the StinkyTofu IR checkpoint editor and say how to open it locally.
#
#   st-ir-editor.sh --repo REPO --bundles DIR --load-dir DIR [--port N]
#
# REPO is the rocm-libraries checkout. The editor and the stinkytofu package
# TensileLite built are both taken from it:
#   <repo>/shared/stinkytofu/tools/stinkytofu-ir-editor
#   <repo>/projects/hipblaslt/tensilelite/build_tmp/tensilelite/rocisa
# Set STINKYTOFU_PYTHON_PATH to use a different directory for the package.
# --bundles is the dump directory (--dump-st-ir); --load-dir is where Save writes.
set -euo pipefail

usage() {
    echo "usage: $(basename "$0") --repo REPO --bundles DIR --load-dir DIR [--port N]" >&2
}

repo=""
bundles=""
load_dir=""
port=8765
port_given=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo) repo=$2; shift 2 ;;
        --bundles) bundles=$2; shift 2 ;;
        --load-dir) load_dir=$2; shift 2 ;;
        --port) port=$2; port_given=1; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done
if [[ -z $repo || -z $bundles || -z $load_dir ]]; then
    usage
    exit 1
fi

repo=$(cd "$repo" && pwd)
if [[ ! -d $bundles ]]; then
    echo "no dump directory: $bundles" >&2
    exit 1
fi
bundles=$(cd "$bundles" && pwd)
mkdir -p "$load_dir"
load_dir=$(cd "$load_dir" && pwd)
if [[ $bundles == "$load_dir" ]]; then
    echo "the load directory must differ from the bundles directory" >&2
    exit 1
fi

editor=$repo/shared/stinkytofu/tools/stinkytofu-ir-editor
if [[ ! -f $editor/stinkytofu_ir_editor/__main__.py ]]; then
    echo "no IR editor in $repo" >&2
    exit 1
fi
if [[ -n ${STINKYTOFU_PYTHON_PATH:-} ]]; then
    st_path=$STINKYTOFU_PYTHON_PATH
else
    st_path=$repo/projects/hipblaslt/tensilelite/build_tmp/tensilelite/rocisa
fi
if [[ ! -d $st_path/stinkytofu ]]; then
    echo "stinkytofu package not found in $st_path" >&2
    echo "build TensileLite, or set STINKYTOFU_PYTHON_PATH to the directory that contains it" >&2
    exit 1
fi

if [[ -x $HOME/pyenv/bin/python ]]; then
    python=$HOME/pyenv/bin/python
else
    python=python3
fi

# First port >= $1 that this machine can bind on 127.0.0.1, within 100 ports.
free_port() {
    "$python" - "$1" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for candidate in range(start, start + 100):
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", candidate))
    except OSError:
        continue
    finally:
        sock.close()
    print(candidate)
    break
else:
    sys.exit("no free port from %s through %s" % (start, start + 99))
PY
}

if [[ $port_given -eq 0 ]]; then
    chosen=$(free_port "$port")
    if [[ $chosen -ne $port ]]; then
        echo "port $port is in use; using $chosen"
    fi
    port=$chosen
else
    chosen=$(free_port "$port")
    if [[ $chosen -ne $port ]]; then
        echo "port $port is already in use. Stop that server, or omit --port." >&2
        exit 1
    fi
fi

cd "$editor"
"$python" -m stinkytofu_ir_editor \
    --bundles "$bundles" \
    --load-dir "$load_dir" \
    --stinkytofu-path "$st_path" \
    --host 127.0.0.1 \
    --port "$port" &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT INT TERM

ready=0
for _ in $(seq 1 50); do
    if curl -sf -o /dev/null "http://127.0.0.1:$port/"; then
        ready=1
        break
    fi
    if ! kill -0 "$server" 2>/dev/null; then
        wait "$server" || true
        exit 1
    fi
    sleep 0.1
done
if [[ $ready -ne 1 ]]; then
    echo "the editor did not start on port $port" >&2
    exit 1
fi

url="http://127.0.0.1:$port/"
if [[ -n ${SSH_CONNECTION:-} || -n ${SSH_TTY:-} ]]; then
    host=$(hostname -f 2>/dev/null || hostname)
    echo
    echo "This server is only reachable on ${host} itself."
    echo "On your own machine, run this. It picks a free local port, prints the"
    echo "address to open, and forwards that port to $port here:"
    echo
    cat <<EOF
remote=$port
host=$(whoami)@$host
local=\$remote
while bash -c "echo >/dev/tcp/127.0.0.1/\$local" >/dev/null 2>&1; do
    local=\$((local + 1))
    if (( local > remote + 50 )); then
        echo "no free local port from \$remote" >&2
        exit 1
    fi
done
echo "open http://127.0.0.1:\$local/"
exec ssh -N -o ExitOnForwardFailure=yes -L "\$local:127.0.0.1:\$remote" "\$host"
EOF
    echo
    echo "Leave it running, then open the address it prints."
elif [[ -n ${DISPLAY:-} ]] && command -v xdg-open >/dev/null; then
    xdg-open "$url" >/dev/null 2>&1 || true
fi

wait "$server"
