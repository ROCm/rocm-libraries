#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Rebuild the cross-engine evaluation integration branch from the remote tips of its inputs
# (integration_inputs.txt), so an update to any input is picked up by re-running this script.
#
#   rebuild_integration.sh [--remote hub] [--branch users/jascampb/uhd-xengine-eval-integration]
#                          [--base <ref>] [--inputs <file>] [--continue]
#
# Run it from the integration worktree with the branch checked out and no tracked changes.
# The branch is reset to the base and every input merged with --no-ff, in file order. The base
# defaults to the newest merge-base of the inputs with <remote>/develop, so no develop commit that
# none of the inputs was built on is pulled in; pass --base <remote>/develop to move it forward.
#
# Conflicts: git rerere is enabled and replays every resolution recorded earlier. A conflict it
# cannot resolve stops the script with the paths listed: resolve, `git commit`, then re-run with
# --continue, which merges only the inputs HEAD does not contain yet. rerere records that
# resolution for the next rebuild.
#
# The branch is rewritten, so publishing it is a force push; the script prints the command and
# never pushes.
set -euo pipefail

main() {
    local remote=hub
    local branch=users/jascampb/uhd-xengine-eval-integration
    local base=""
    local resume=0
    local inputs_file
    inputs_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/integration_inputs.txt"

    while (($#)); do
        case "$1" in
        --remote) remote="$2"; shift 2 ;;
        --branch) branch="$2"; shift 2 ;;
        --base) base="$2"; shift 2 ;;
        --inputs) inputs_file="$2"; shift 2 ;;
        --continue) resume=1; shift ;;
        -h | --help) sed -n '4,24p' "${BASH_SOURCE[0]}"; return 0 ;;
        *) echo "unknown argument: $1" >&2; return 2 ;;
        esac
    done

    local inputs=()
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line//[[:space:]]/}"
        [[ -n "$line" ]] && inputs+=("$line")
    done <"$inputs_file"
    ((${#inputs[@]})) || { echo "no inputs in $inputs_file" >&2; return 2; }

    local current
    current="$(git symbolic-ref --quiet --short HEAD || true)"
    if [[ "$current" != "$branch" ]]; then
        echo "check out $branch in this worktree first (HEAD is ${current:-detached})" >&2
        return 2
    fi
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "tracked changes in the worktree; commit or stash them first" >&2
        return 2
    fi
    if [[ -e "$(git rev-parse --git-path MERGE_HEAD)" ]]; then
        echo "a merge is in progress; finish it (git commit) or abort it first" >&2
        return 2
    fi

    local refspecs=("+refs/heads/develop:refs/remotes/$remote/develop")
    local input
    for input in "${inputs[@]}"; do
        refspecs+=("+refs/heads/$input:refs/remotes/$remote/$input")
    done
    git fetch --no-tags "$remote" "${refspecs[@]}"

    git config rerere.enabled true
    git config rerere.autoupdate true

    if ((!resume)); then
        if [[ -z "$base" ]]; then
            base="$(newest_merge_base "$remote" "${inputs[@]}")"
        fi
        echo "previous tip of $branch: $(git rev-parse HEAD)  (recover with: git reset --hard <that sha>)"
        echo "base: $(git log -1 --format='%h %cs %s' "$base")"
        git reset --quiet --hard "$base"
    fi

    for input in "${inputs[@]}"; do
        local ref="$remote/$input"
        if git merge-base --is-ancestor "$ref" HEAD; then
            echo "contained: $ref @ $(git rev-parse --short "$ref")"
            continue
        fi
        echo "merging:   $ref @ $(git rev-parse --short "$ref")"
        if ! git merge --quiet --no-ff --no-edit "$ref"; then
            local unresolved
            unresolved="$(git diff --name-only --diff-filter=U)"
            if [[ -n "$unresolved" ]]; then
                echo >&2
                echo "unresolved conflicts merging $ref:" >&2
                echo "$unresolved" | sed 's/^/  /' >&2
                echo "resolve them, git add + git commit, then re-run with --continue" >&2
                return 1
            fi
            # rerere resolved and staged every conflict: record the merge.
            git commit --quiet --no-edit
            echo "           (conflicts resolved from recorded resolutions)"
        fi
    done

    echo
    echo "integration: $(git log -1 --format='%h' HEAD) on base $(git rev-parse --short "${base:-HEAD}")"
    for input in "${inputs[@]}"; do
        printf '  %-55s %s\n' "$input" "$(git rev-parse --short "$remote/$input")"
    done
    echo "publish with: git push --force-with-lease $remote $branch"
}

# The newest of the inputs' merge-bases with develop: every other one must be its ancestor.
newest_merge_base() {
    local remote="$1"
    shift
    local bases=() input candidate other ok
    for input in "$@"; do
        bases+=("$(git merge-base "$remote/$input" "$remote/develop")")
    done
    for candidate in "${bases[@]}"; do
        ok=1
        for other in "${bases[@]}"; do
            git merge-base --is-ancestor "$other" "$candidate" || { ok=0; break; }
        done
        if ((ok)); then
            echo "$candidate"
            return 0
        fi
    done
    echo "the inputs' develop merge-bases are not on one line; pass --base" >&2
    return 1
}

# Parsed whole before it runs: the reset above rewrites this file on disk.
main "$@"
exit $?
