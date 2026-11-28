#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The container's entry point (tini runs it as PID 1's child, so signals reach it). On the host, docker/rkmon runs each
# of these for you (rkmon login github_write, rkmon rk status, rkmon shell, ...; rkmon help).
#
#   run                     the default: the monitor service behind the admin dashboard, the tick, and, once Claude is
#                           set up, the orchestrator's long-lived Claude session (rk orchestrate), whose output is
#                           the container's log too (docker/rkmon logs). Each is restarted when it dies, except after
#                           an rk stop or rk kill: then they stay down until rk start (rk stopped says which).
#                           `docker stop` stops them all (rk stop), and the container's restart policy, not systemd,
#                           brings them back (its start runs rk start): the orchestrator then resumes its session
#   rk ARGS                 any rk command (rk doctor, rk status, rk features, ...); as the writer, only the writer's
#                           commands, through its broker (docker/rk_write.py) without sudo: rkmon secret NAME sets a
#                           secret the writer keeps so, and rkmon web-host install installs the web host's request
#                           service so, from the host
#   atlassian ARGS          the atlassian CLI under its login, as the writer only (rkmon login atlassian runs
#                           `atlassian auth login` as the write user: the login goes to /secrets-write/atlassian)
#   claude ARGS             the Claude Code CLI with this deployment's settings (claude --version, ...)
#   gh ARGS                 the GitHub CLI with this deployment's settings: the read login (auth login, auth status, ...)
#   gh-write ARGS           the GitHub CLI under the write login, as the write user only (rkmon login github_write
#                           runs `gh-write auth login` as the write user: the login goes to /secrets-write/gh, which
#                           no other user reads)
#   code-install            as root (RKMON_AS=root): the package's code, a tar archive on stdin, installed in the code
#                           volume as a release over this image (rkmon apply; rkmon start too, with the option that
#                           leaves the code kept as it is: docker/rk_root.py)
#   code-back               as root (RKMON_AS=root): the code kept before the last install runs again (rkmon rollback)
#   shell                   a shell, as the monitor's user
#   health                  the image's HEALTHCHECK: unhealthy (exit 1) while a piece crash-loops (HEALTH_RESTARTS
#                           restarts or more in the last 30 minutes), or while whether the work volume is set up
#                           cannot be told (the line says why); healthy while it is not set up yet or stopped
#
# gh's read login always goes to the secrets volume (GH_CONFIG_DIR /secrets/gh-read), also before a start: a login in
# a one-off container's home would be thrown away with it. The orchestrator's Claude session gets the same folder
# (rkmon_lib/orchestrator.py, session_env), so its own gh calls run logged in, and only read. The atlassian CLI's login
# is the writer's alone (/secrets-write/atlassian), and so is the GitHub write login
# (/secrets-write/gh): rk's GitHub write commands run as that user through the write broker (docker/rk_write.py, one
# sudo rule), with code no user can change, which it checks first. The code of pull requests (the checks' builds and
# tests, the probes) runs as a PR runner (rkmon_lib/container_layout.py), in its slot, through its launcher
# (docker/rk_prcode.py, one sudo rule): it reads neither secrets volume nor the monitor's state, and writes only its
# slot's folders (in /ramdisk/slots). The root step of each start of `run` (docker/rk_root.py start, in the root
# phase below; no sudo rule names it) moves a write login in /secrets/gh to the write user's volume, makes the one-time
# move's copies of the secrets whose name or keeper changed, and keeps the code volume root's.
#
# Everything here runs with umask 027. What the monitor and the writer share (the state, the logs, the outputs) gets
# files 0660 and folders 2770, group rkmon, from its folder's default ACL; the trees PR code reads (the clone, the
# worktrees, the worker tools, the code) are the runners' group's, read only; the slots are the runners'. As root,
# right after the ids, each start gives the working folder and /ramdisk these permissions and makes the slots in
# /ramdisk (docker/rk_layout.py), and the start of `run` keeps the clone's own files that way
# (core.sharedRepository=0640).
#
# The code: the image's (/opt/rocke_monitor), or the release root installed over it in the code volume (rkmon apply
# runs code-install: /code/current, rkmon_lib/code_volume.py), root's and read-only to every other user. Each piece
# starts from the one `rk --which-code` names, at each restart, with RKMON_CODE naming it, so every rk the piece starts
# runs that code too. At each start the worker tools are installed again from it (rk worker-tools --install), so an
# install or a new image never leaves the reviews on older worker scripts. While the root step fails (the code volume
# is not root's and read-only all the way, or cannot be checked; the write login's move fails; or a copy of the
# one-time move cannot be read or written), the start of `run` says what is wrong and its fix, and waits: the monitor
# does not start. A failed check also removes /code/current, so no release runs meanwhile: the health check, and any
# rk, run the image's own code.
#
# The logs in /work/logs are cut at config loop_log_max_mb (the tick loop's limit, `rk loop-config`): a log past it is
# copied to its name with .1 added and emptied in place, at the start and at each round of the tick loop that runs
# without the orchestrator, so a writer keeps its file. While the configuration cannot be read, no log is cut.
#
# The container starts as root, with only the capabilities its compose file keeps (rkmon_lib/container_layout.py
# CAPABILITIES). Its root phase first gives the monitor and its group the deployment's ids (docker/rk_ids.py:
# RKMON_UID, RKMON_GID, from docker/.env; the writer and the PR runners keep the package's ids), then the working
# folder's permissions and the slots (docker/rk_layout.py), then, for `run`, the root step (docker/rk_root.py start).
# An rk command handed your confirmed settings (RKMON_CONFIRMED_LAYER, below) is the logins' walk's or rkmon web-host
# install's (docker/rkmon walk_inside), in a one-off container of the image you confirmed, beside the running
# container, which may run another image: its root phase gives the users their ids alone (docker/rk_ids.py --users),
# so that it changes no owner or permission the running container uses, and that container's own /tmp and /var/tmp
# the start's permissions (docker/rk_layout.py --system-tmp: the command's temporary folders go there), then writes
# those settings where rk reads them (docker/rk_root.py hand-layer). A command but `run` with RKMON_LEAVE_VOLUMES set
# (below) gets the first two of these steps alone, so that it walks no file of the volumes and writes no ids record.
# Then the command runs as the user RKMON_AS names: the monitor (the default); the writer, for its logins and secrets
# and for rkmon web-host install, from the host; or root, for code-install, code-back and the image test's root
# stages. The health check drops to the monitor without these steps, and code-install and code-back, which change the
# code volume alone, run without them. In the running container docker/rkmon runs this file as that user itself (exec
# --user), and no step is needed.
#
# Environment: RKMON_PORT and RKMON_ADMIN_HOST, the host port and address docker/compose.yaml publishes the admin
# dashboard on (README.md, "The admin dashboard"). RKMON_CONFIRMED_LAYER: the layer your confirmed settings make, which
# docker/rkmon hands its one-off containers of the logins' walk and of rkmon web-host install alone (walk_inside), by
# the variable's name on docker's command line. It carries no secret: docker inspect shows it, and tini, the
# container's first process, keeps it for the one-off's run. This file takes it out of its environment first, so the
# command does not inherit it, and as root writes it for an rk command only (rkmon_lib/container_layout.py
# HANDED_LAYER_DIR). RKMON_LEAVE_VOLUMES (container_layout LEAVE_VOLUMES_VARIABLE), set and not empty: the one-off
# container's command only reads (a question, adopt's plan), so its root phase leaves the volumes' files as they are,
# but for a new volume's own folder, which gets the owner a start gives it (docker/rk_ids.py --users, above); this file
# takes it out of its environment first too.
# The minutes between ticks are the configuration's claude.tick_min (`rk tick-min`): your deploy settings'
# monitor.tick_interval_minutes.
set -u
umask 027  # no one outside a user's group reads what it makes; what the users share takes its folder's default ACL
PKG=/opt/rocke_monitor  # the image's code; `code` names the code that runs (a release over it: rkmon apply)
say() {  # each line of its words but an empty one, with the time and the container's name: the container's log
    printf '%s\n' "$*" | while IFS= read -r line; do
        [ -z "$line" ] || printf '%s rkmon-container: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$line"
    done
}
handed=${RKMON_CONFIRMED_LAYER:-}  # unexported: the root phase's alone (Environment, above)
unset RKMON_CONFIRMED_LAYER
leave=${RKMON_LEAVE_VOLUMES:-}  # unexported too
unset RKMON_LEAVE_VOLUMES

if [ "$(id -u)" = 0 ]; then
    as=${RKMON_AS:-rkmon}
    case "$as" in
        rkmon|rkwrite|root) ;;
        *) echo "RKMON_AS is rkmon, rkwrite or root, not $as" >&2; exit 2 ;;
    esac
    walk=
    [ -n "$handed" ] && [ "${1:-run}" = rk ] && walk=1  # a one-off's rk command handed your confirmed settings (above)
    users=$walk  # the users' ids alone: the walk's, and a one-off's that leaves the volumes as they are (above)
    [ -n "$leave" ] && [ "${1:-run}" != run ] && users=1
    case "${1:-run}" in
        health|code-install|code-back) ;;
        *)
            if [ -n "$users" ]; then
                python3 -I "$PKG/docker/rk_ids.py" --users || exit $?
                python3 -I "$PKG/docker/rk_layout.py" --system-tmp || exit $?
            else
                python3 -I "$PKG/docker/rk_ids.py" || exit $?
                python3 -I "$PKG/docker/rk_layout.py" || exit $?
            fi ;;
    esac
    if [ -n "$walk" ]; then  # root's copy of your confirmed settings, which rk reads
        printf '%s' "$handed" | python3 -I "$PKG/docker/rk_root.py" hand-layer || exit $?
    fi
    handed=
    if [ "${1:-run}" = run ]; then  # the root step, at each start of `run`: what it did goes to the log
        said=$(python3 -I "$PKG/docker/rk_root.py" start 2>&1)
        rc=$?
        say "$said"
        if [ "$rc" != 0 ]; then  # nothing starts, and the container waits for the fix
            say "the monitor does not start: the root step failed (exit $rc; the lines above say why); waiting"
            trap 'kill -TERM "$w" 2>/dev/null; exit 0' TERM INT
            while :; do sleep 3600 & w=$!; wait "$w"; done
        fi
    fi
    if [ "$as" != root ]; then
        exec setpriv --reuid="$as" --regid="$(id -g "$as")" --init-groups -- "$0" "$@"
    fi
fi
WRITE_LOGIN=/secrets-write/gh  # the GitHub write login: the write user's only
ATLASSIAN_LOGIN=/secrets-write/atlassian  # the atlassian CLI's login: the writer's only
RK="$PKG/rk"
ORCH_PY=/opt/rkmon/venv-claude/bin/python
# The container's logs: rkmon_lib/container_layout.py WORK_DIR/LOGS_FOLDER (tests/test_container.py), or the folder
# a test names
LOGS=${RKMON_LOGS:-/work/logs}
HEALTH_RESTARTS=5  # restarts of the kept pieces in 30 minutes that make the container unhealthy
mkdir -p "$LOGS" 2>/dev/null

# The monitor's home (its state/) is RKMON_HOME, as for rk: the image's environment names it (docker/Dockerfile:
# rkmon_lib/container_layout.py's MONITOR_HOME), and the tests their own. Without it, the command stops at the first
# provisioned call, naming it (set -u): so that call runs in this shell, with its errors shown.
# Whether the work volume is set up in that home, as rk provision --check says, by its exit (SETUP_RC): 0 set up (a
# volume an older package set up counts), 1 not set up yet, any other when it cannot tell. SETUP_SAID: what it said, a
# line that says so, or names the first start that sets it up, as you run it on the host (rkmon start), or says why it
# cannot tell.
provisioned() {
    : "$RKMON_HOME"
    SETUP_SAID=$(RKMON_HOME=$RKMON_HOME RKMON_WRAPPER=rkmon "$RK" provision --check) && SETUP_RC=0 || SETUP_RC=$?
    if [ "$SETUP_RC" -gt 1 ] && [ -z "$SETUP_SAID" ]; then
        SETUP_SAID="whether the work volume is set up cannot be told (exit $SETUP_RC)"
    fi
    return "$SETUP_RC"
}

code() { "$RK" --which-code 2>/dev/null || echo "$PKG"; }  # read again at each (re)start: an install may change it

with_settings() {  # the tool environment rk gives its own commands (gh), for a command run by hand
    envline=$("$RK" env 2>/dev/null) || envline=""  # no configuration yet: only the read login's folder below
    case "$envline" in "export "*) eval "$envline" ;; esac
    secrets=${RKMON_SECRETS_DIR:-/secrets}
    GH_CONFIG_DIR=${GH_CONFIG_DIR:-$secrets/gh-read}
    export GH_CONFIG_DIR
}

cmd=${1:-run}
[ $# -gt 0 ] && shift
case "$cmd" in
    rk)
        if [ "$(id -un)" = rkwrite ]; then  # the write user's rk is its broker's, with its checks, as sudo runs it
            exec "$PKG/docker/rk_write.py" "$@"
        fi
        exec "$RK" "$@" ;;
    claude|gh)
        with_settings
        exec "$cmd" "$@" ;;
    atlassian)
        if [ "$(id -un)" != rkwrite ]; then
            echo "atlassian: its login is the writer's only (rkmon login atlassian runs it as the writer)" >&2
            exit 2
        fi
        mkdir -p "$ATLASSIAN_LOGIN" && chmod 700 "$(dirname "$ATLASSIAN_LOGIN")" "$ATLASSIAN_LOGIN"
        ATLASSIAN_CONFIG_DIR=$ATLASSIAN_LOGIN
        export ATLASSIAN_CONFIG_DIR
        exec atlassian "$@" ;;
    gh-write)
        if [ "$(id -un)" != rkwrite ]; then
            echo "gh-write: the write login is the write user's only (rkmon login github_write runs it as the writer)" >&2
            exit 2
        fi
        mkdir -p "$WRITE_LOGIN" && chmod 700 "$(dirname "$WRITE_LOGIN")" "$WRITE_LOGIN"
        unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GITHUB_ENTERPRISE_TOKEN
        GH_CONFIG_DIR=$WRITE_LOGIN
        export GH_CONFIG_DIR
        exec gh "$@" ;;
    code-install|code-back)  # root installs the code, or goes back to the code kept (rkmon apply, rkmon rollback)
        if [ "$(id -u)" != 0 ]; then echo "$cmd: as root only (rkmon apply and rkmon rollback run it)" >&2; exit 2; fi
        exec python3 -I "$PKG/docker/rk_root.py" "$cmd" "$@" ;;
    shell|bash|sh)
        with_settings
        exec /bin/bash "$@" ;;
    health)
        provisioned
        case $SETUP_RC in
            0) ;;
            1) echo "healthy: not set up yet"; exit 0 ;;
            *) printf 'unhealthy: %s\n' "$(printf '%s\n' "$SETUP_SAID" | sed -n '/./{p;q;}')"; exit 1 ;;
        esac
        if "$RK" stopped >/dev/null 2>&1; then echo "healthy: stopped (rk start resumes)"; exit 0; fi
        since=$(( $(date +%s) - 1800 ))
        n=$(awk -v t="$since" '$1 >= t {n++} END {print n+0}' "$LOGS/restarts" 2>/dev/null)
        if [ "${n:-0}" -ge "$HEALTH_RESTARTS" ]; then
            last=$(awk -v t="$since" '$1 >= t {$1 = ""; last = $0} END {print last}' "$LOGS/restarts")
            echo "unhealthy: $n restarts in the last 30 minutes (the last:$last); see $LOGS"
            exit 1
        fi
        echo "healthy"
        exit 0 ;;
    run) ;;
    *)
        echo "usage: run | rk ARGS | atlassian ARGS | claude ARGS | gh ARGS | gh-write ARGS" \
             "| code-install | code-back | shell | health" >&2
        exit 2 ;;
esac

# run --------------------------------------------------------------------------------------------------------------
read_login_note() {  # while there is no read login, the one step that adds it. Its file in its folder: the secrets
                     # registry's github_read and rkmon_lib/gh.py HOSTS_FILE (tests/test_container.py)
    if [ ! -f "${RKMON_SECRETS_DIR:-/secrets}/gh-read/hosts.yml" ]; then
        say "no GitHub read login yet: run   docker/rkmon login github_read   on the host, once (a read-only token," \
            "the repository read only); the monitor reads GitHub with it, and the write login is the write user's only"
    fi
}
read_login_note

wait_until_set_up() {  # nothing runs while the work volume is not set up: what rk provision --check says (not set up
                       # yet, or why it cannot tell), again only when it says something else, and a look every 30 s
    provisioned && return 0
    said=""
    trap 'kill -TERM "$w" 2>/dev/null; exit 0' TERM INT
    while :; do
        if [ "$SETUP_SAID" != "$said" ]; then
            say "$SETUP_SAID"
            [ -n "$said" ] || say "waiting: the monitor starts once the work volume is set up"
            said=$SETUP_SAID
        fi
        sleep 30 & w=$!; wait "$w"
        provisioned && break
    done
    trap - TERM INT
    say "set up: starting"
}
wait_until_set_up

loop_value() {  # loop_value NAME: the tick loop's setting NAME, a whole number, from `rk loop-config` (the
                # configuration's: no value of its own here); "" while the configuration cannot be read
    "$RK" loop-config 2>/dev/null | sed -n "s/^$1='\{0,1\}\([0-9]*\).*/\1/p"
}
LOG_MAX=$(loop_value LOOP_LOG_MAX_BYTES)
shared_repository() {  # the clone's own files (.git): the runners' group reads them, and no one else
    (
        envline=$("$RK" env 2>/dev/null) || exit 0
        case "$envline" in "export "*) eval "$envline" ;; *) exit 0 ;; esac
        [ -d "${RKMON_REPO_DIR:-}/.git" ] || exit 0
        git -C "$RKMON_REPO_DIR" config core.sharedRepository 0640
    )
}
shared_repository

rotate_logs() {  # a log past LOG_MAX: copied to its name with .1 added and emptied in place (its writers keep their
                 # file). While there is no LOG_MAX (the configuration could not be read), each call reads it again and
                 # cuts none
    [ -n "$LOG_MAX" ] || LOG_MAX=$(loop_value LOOP_LOG_MAX_BYTES)
    for f in "$LOGS"/*.log; do
        [ -n "$LOG_MAX" ] && [ -f "$f" ] || continue
        size=$(wc -c <"$f" 2>/dev/null) || continue
        if [ "${size:-0}" -gt "$LOG_MAX" ]; then
            cp "$f" "$f.1" 2>/dev/null && : >"$f"
        fi
    done
    if [ -f "$LOGS/restarts" ] && [ "$(wc -l <"$LOGS/restarts")" -gt 1000 ]; then  # the health record: its tail
        tail -n 200 "$LOGS/restarts" >"$LOGS/restarts.new" && mv "$LOGS/restarts.new" "$LOGS/restarts"
    fi
}
rotate_logs

STATE=$RKMON_HOME/state  # the monitor's state: a stop waits for its write lock
pids=""
sleeper=""
stopping=0
stop_all() {
    stopping=1
    trap '' TERM INT
    say "stopping: the orchestrator, then rk stop (the review driver, the work in flight, the dashboard service)"
    for p in $pids; do kill -TERM "$p" 2>/dev/null; done
    "$RK" stop >>"$LOGS/stop.log" 2>&1
    # A GitHub write rk stop left alone to finish (one the session started shares this process group) ends before
    # the group is signalled: wait for the write lock, a bounded time (within compose's stop_grace_period).
    if [ -f "$STATE/write.lock" ] && command -v flock >/dev/null 2>&1; then
        flock -w 120 "$STATE/write.lock" true 2>/dev/null ||
            say "an outward write still held the write lock after 120 s: stopping anyway"
    fi
    # Then whatever is left of this process group (a pause between ticks or restarts): nothing outlives the stop,
    # and nothing keeps the output open after it.
    [ -n "$sleeper" ] && kill -TERM "$sleeper" 2>/dev/null
    kill -TERM 0 2>/dev/null
    for p in $pids; do wait "$p" 2>/dev/null; done
    say "stopped"
    exit 0
}
trap stop_all TERM INT

is_stopped() { "$RK" stopped >/dev/null 2>&1; }  # after an rk stop or rk kill: nothing is started again until rk start

keep() {  # keep NAME LOG CMD...: run CMD, and run it again when it ends (a backoff of up to 5 minutes), until the
          # container stops; after an rk stop or rk kill, it stays down until rk start. Each end is recorded in
          # $LOGS/restarts (the health check counts them)
    name=$1; log=$2; shift 2
    (
        child=""
        trap '[ -n "$child" ] && kill -TERM "$child" 2>/dev/null; wait 2>/dev/null; exit 0' TERM INT
        delay=10
        while :; do
            if is_stopped; then
                say "$name stays down: the monitor is stopped (rk start resumes it)" >>"$log"
                while is_stopped; do sleep 30 & child=$!; wait "$child"; done
                child=""
                say "$name: rk start ran; starting it again" >>"$log"
                delay=10
            fi
            started=$(date +%s)
            "$@" >>"$log" 2>&1 &
            child=$!
            wait "$child"
            rc=$?
            ran=$(( $(date +%s) - started ))
            [ "$ran" -gt 600 ] && delay=10
            say "$name ended (exit $rc after ${ran}s); again in ${delay}s" >>"$log"
            [ -n "${LOGS:-}" ] && echo "$(date +%s) $name (exit $rc)" >>"$LOGS/restarts" 2>/dev/null
            sleep "$delay"
            delay=$(( delay * 2 )); [ "$delay" -gt 300 ] && delay=300
        done
    ) &
    pids="$pids $!"
}

# rk start undoes an earlier rk stop or kill (a container stop runs rk stop) and starts the dashboard service
# (tools/rkweb.py: rk serve, which in the image listens on every address and answers for the published port). What it
# says, the one-time move's report with it, goes to the start's log and to the container's log (docker/rkmon logs).
said=$("$RK" start 2>&1)
printf '%s\n' "$said" >>"$LOGS/start.log"
say "$said"
# The worker tools from the code that runs now (after a new image the old copies would stay).
"$RK" worker-tools --install >>"$LOGS/start.log" 2>&1 ||
    say "the worker tools were not installed again (rk worker-tools --install; $LOGS/start.log says why)"

# The deployment's features say which parts run: rk features --runs PART exits 0 for a part it runs and 1 for one it
# does not. While the configuration cannot be read (it exits 4, and rk tick-min fails), no part starts and no tick
# runs; rk doctor shows the FAIL and its fix.
runs() { "$RK" features --runs "$1" >/dev/null 2>&1; }
unreadable() { "$RK" features --runs web >/dev/null 2>&1; [ $? -eq 4 ]; }
dashboard() {  # the dashboard service, started again whenever it is down, but not while stopped (rk stop or rk kill)
    while :; do
        c=$(code)  # the code the service runs, and every rk it starts (RKMON_CODE)
        if runs web; then
            is_stopped || RKMON_CODE=$c "$c/tools/rkweb.py" status 2>/dev/null | grep -q "^RKWEB running" ||
                RKMON_CODE=$c "$c/tools/rkweb.py" >/dev/null 2>&1
        fi
        sleep 60
    done
}
keep "the loop that restarts the admin dashboard" "$LOGS/web.log" dashboard

# The orchestrator: a long-lived Claude session that runs the skill's tick (it runs rk tick itself). It is kept
# whether or not it is set up when the container starts: before each (re)start it waits until `rk orchestrate --check`
# passes (the Claude login, made later with rkmon login claude, takes effect without a restart). Without it, the tick
# loop below runs the toolkit's part alone, and the board says that Claude's part waits.
orch_ready() { "$RK" orchestrate --check >/dev/null 2>&1; }
orchestrator() {
    if unreadable; then
        say "the configuration cannot be read, so no Claude session starts (rk doctor says why and how to fix it)"
        while unreadable; do sleep "${RKMON_ORCH_WAIT_S:-60}"; done
        say "the configuration can be read"
    fi
    if ! runs claude; then  # no Claude session: the tick loop below runs each tick by fixed rules (rk rules-tick)
        say "review automation is off, so no Claude session runs: each tick is run by fixed rules, and what needs judgement waits for a person (rk status)"
        while ! runs claude; do sleep "${RKMON_ORCH_WAIT_S:-60}"; done
        say "review automation is on now: starting the orchestrator"
    fi
    if ! orch_ready; then
        say "the orchestrator is not set up yet (rk orchestrate --check says why); it starts once it is"
        "$RK" orchestrate --check
        while ! orch_ready; do sleep "${RKMON_ORCH_WAIT_S:-60}"; done
        say "the orchestrator is set up: starting it"
    fi
    RKMON_CODE=$(code)  # the code the session runs, and every rk it starts
    export RKMON_CODE
    exec "$ORCH_PY" -B "$RKMON_CODE/rkmon.py" orchestrate
}
keep "the orchestrator" "$LOGS/orchestrator.log" orchestrator
# Its lean output (a line per tick, the !! lines, what went to the session) is the container's log as well.
tail -n 0 -F "$LOGS/orchestrator.log" 2>/dev/null &
pids="$pids $!"

# The tick loop that runs without the orchestrator: every claude.tick_min minutes (rk tick-min) rk tick runs while the
# orchestrator is not set up, or when its own tick is overdue (rk tick-due says), unless the monitor is stopped
# (rk stop or rk kill). It waits between its ticks with rk tick-wait, which marks the loop for rk tick-now and takes
# its request: a request it takes starts a tick at once. When rk tick-wait ends in any other way before the wait is
# over (it did not run, or it failed), the loop sleeps the rest of the wait. While the configuration cannot be read
# (rk tick-min fails), no tick runs, and the loop looks again every minute. Each tick has the tick loop's time limit
# (config tick_timeout_min, `rk loop-config`): past it, timeout ends the tick's whole process group, so no hung tick
# keeps the state lock. Without a limit (the configuration could not be read just then), no tick runs that round, and
# a request taken waits for the next. The loop itself is kept like the others (it starts again if it dies).
ticks_without_orchestrator() {
    # this loop's own pid, for its mark: $$ would name the entry point (keep runs the function with &)
    me=$(exec sh -c 'echo "$PPID"')
    asked=""  # 1 while a request rk tick-wait took waits for its tick
    "$RK" tick-wait --loop "$me" 0 && asked=1  # marks the loop at once, and takes a request already there
    waiting=""
    while :; do
        if ! TICK_MIN=$("$RK" tick-min 2>/dev/null); then
            if [ -z "$waiting" ]; then
                say "no tick runs: the configuration cannot be read (rk doctor says why and how to fix it)"
            fi
            waiting=1
            rotate_logs
            sleep 60
            continue
        fi
        [ -n "$waiting" ] && say "the configuration can be read: the ticks run again"
        waiting=""
        if is_stopped; then
            :
        elif [ -n "$asked" ] || ! orch_ready || "$RK" tick-due --minutes "$((TICK_MIN * 2 + 5))" >/dev/null 2>&1; then
            limit=$(loop_value TICK_TIMEOUT_S)
            if [ -z "$limit" ]; then
                say "no tick ran: its time limit could not be read from the configuration (rk doctor says why and" \
                    "how to fix it); the next round asks again"
            else
                asked=""
                if "$RK" features --runs rules >/dev/null 2>&1; then  # no Claude session: the tick, run by fixed rules
                    timeout -k 30 "$((limit + 600))" "$RK" rules-tick
                else
                    timeout -k 30 "$limit" "$RK" tick
                fi
            fi
        fi
        rotate_logs
        waited=$(date +%s)
        if "$RK" tick-wait --loop "$me" "$((TICK_MIN * 60))"; then
            asked=1  # rk tick-now's request, taken: a tick at once
        else
            left=$((TICK_MIN * 60 - ($(date +%s) - waited)))
            [ "$left" -le 0 ] || sleep "$left"
        fi
    done
}
keep "the tick loop that runs without the orchestrator" "$LOGS/tick.log" ticks_without_orchestrator

# What runs, as the features say (rk features --runs): nothing while the configuration cannot be read.
if TICK_MIN=$("$RK" tick-min 2>/dev/null); then
    parts=""
    runs web && parts="the dashboard service, "
    if runs claude; then
        parts="${parts}the orchestrator (once it is set up), the tick every $TICK_MIN min while the orchestrator is not set up or is late"
    else
        parts="${parts}the tick every $TICK_MIN min, run by fixed rules"
    fi
    say "running: $parts"
else
    say "nothing runs while the configuration cannot be read (rk doctor says why and how to fix it)"
fi
while [ "$stopping" -eq 0 ]; do
    sleep 3600 &
    sleeper=$!
    wait "$sleeper"
done
