# Lane: leak sweep (`LEAK`)

Find anything the push or PR publishes that should not be public (this is a public repository). Work in `$SCRATCH/LEAK/` (`$L`).

**Sweep dispatch.** A fresh agent re-dispatched for the output sweep does only *Sweeping review output* (below), after defining
`L`, `meta`, `search`, `scan` and the pattern and term files from steps 1 and 2 (`. "$L/lib.sh"` restores the functions; the
files are in `$L`). It writes `$EVIDENCE/r<N>/lanes/LEAK-sweep.md`: the standard lane report ([../findings.md](../findings.md) *Lane report*)
whose Checked-no-finding states `Sweep: clean`, or whose Findings list each counted hit as `<file>:<line>`.

`$TERMS` is the `Leak terms:` path in `PIN.md`, empty when `absent`; `<owner>/<repo>`, `<pr>` and `<N>` (the round) come from `PIN.md` and the dispatch.

## 1. Collect what is published

| Item | Source |
|---|---|
| Every commit (message, author, committer, added lines); the final diff (added lines with line numbers at `$HEAD`, changed paths) | `history` and `git diff -U0`, below |
| PR title, description, reviews, comments, inline threads | `pr/` in the round directory (PR targets only): `branch.txt`, `pr-text.txt`, `pr-comment.txt`, `pr-inline.txt`. No PR: the commit messages are the future description, and `intent.md` stands in for it |
| Earlier heads | Round N > 1: `history "$BASE..<previous head>"` too, the head in the previous round's `PIN.md` |

`history` covers every commit: a leak removed later is still published. `--diff-merges=remerge` (git 2.36+; older `--cc`) includes
what merge commits introduce. `$L/pub.content` holds one published line per line; the parallel `$L/pub.anchor` holds
`<anchor><TAB><base path>`: `path:line` at `$HEAD` (final diff), `<sha>:<path>:<line>` (commit), `commit <sha>` (message or
identity), or `branch:`, `pr-text:`, `pr-comment:`, `pr-inline:`, `path:` plus a line number. The base path is the file's path at
`$BASE`, or `-`.

```bash
. ./env.sh; L="$SCRATCH/LEAK"; mkdir -p "$L"; : > "$L/pub.anchor"; : > "$L/pub.content"
cat > "$L/lib.sh" <<'EOF'
records() {   # stdin: a -U0 diff or history stream
  awk -v A="$L/pub.anchor" -v C="$L/pub.content" '
    /^\001/ { pre = substr($0, 2, 8); mode = "m"; next }
    /^diff --git / { mode = "h"; old = new = ""; next }
    mode == "m" { print "commit " pre "\t-" >> A; print >> C; next }
    mode == "h" && /^--- / { old = ($2 == "/dev/null") ? "" : substr($0, 7); next }
    mode == "h" && /^\+\+\+ / { new = substr($0, 7); next }
    /^@@ / { mode = "b"; match($0, /\+[0-9]+/); n = substr($0, RSTART + 1, RLENGTH - 1) - 1; next }
    mode == "b" && /^\+/ { n++; print (pre ? pre ":" : "") new ":" n "\t" (old ? old : "-") >> A
                           print substr($0, 2) >> C }'
}
meta() {   # stdin: plain lines, anchored LABEL:LINE
  awk -v lab="$1" -v A="$L/pub.anchor" -v C="$L/pub.content" \
    '{ print lab ":" NR "\t-" >> A; print >> C }'
}
history() {
  git -C "$REPO" log -p -U0 --no-color --src-prefix=a/ --dst-prefix=b/ --diff-merges=remerge \
    --format='%x01%H%n%an <%ae>%n%cn <%ce>%n%B' "$1" | records
}
EOF
. "$L/lib.sh"
history "$BASE..$HEAD"
git -C "$REPO" diff -U0 --no-color --src-prefix=a/ --dst-prefix=b/ "$BASE" "$HEAD" | records
git -C "$REPO" diff --name-only "$BASE" "$HEAD" | meta path
# Branch and PR text. pr/ exists only for a PR target; otherwise intent.md and a branch:<name> target's name stand in.
if [ -d pr ]; then
  for k in branch pr-text pr-comment pr-inline; do meta "$k" < "pr/$k.txt"; done
else
  [ -f intent.md ] && meta intent < intent.md
  BRANCH_NAME=$(sed -n 's/^Change: *//p' PIN.md)   # a range target has no branch
  case $BRANCH_NAME in branch:*) printf '%s\n' "${BRANCH_NAME#branch:}" | meta branch ;; esac
fi
```

## 2. Search

Run all three; record every pattern file and the inputs present, findings or not. **Baseline patterns**: save as `$L/patterns.txt`, one case-sensitive extended regex per line;
each matches a whole token (path, host, address, URL):

```text
-----BEGIN [A-Z ]*PRIVATE KEY
(ghp|gho|ghs|ghu|ghr|github_pat)_[A-Za-z0-9_]{20,}
AKIA[0-9A-Z]{16}
xox[abprs]-[A-Za-z0-9-]+
([Aa][Pp][Ii][_-]?[Kk][Ee][Yy]|[Ss][Ee][Cc][Rr][Ee][Tt]|[Tt][Oo][Kk][Ee][Nn]|[Pp][Aa][Ss][Ss]([Ww][Oo][Rr])?[Dd])["']?[[:space:]]*[:=][[:space:]]*["']?[^[:space:]"',;]+
[a-z][a-z0-9+.-]*://[^/[:space:]:@]+:[^/[:space:]@]+@
(^|[^0-9.])(10\.[0-9]+|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+
([a-z0-9-]+\.)+(local|lan|corp|internal|intranet|intra)((\.[a-z0-9-]+)+|[^a-z0-9-]|$)
[^[:space:]"'`()<>]*(sbatch|srun|salloc|squeue|#SBATCH|--partition|qsub|bsub|enroot|apptainer|singularity)[^[:space:]"'`()<>]*
(^|[[:space:]"'`(=:])(/home|/Users|/var/tmp|/scratch|/nfs|/mnt|/tmp|/cluster|/lustre|/data)/[^[:space:]"'`)>]+
(^|[[:space:]"'`(=:])~/[^[:space:]"'`)>]+
\$HOME/[^[:space:]"'`)>]+
(^|[^A-Za-z0-9])[A-Za-z]:[\\/][^[:space:]"'`)>]+
(^|[[:space:]"'`(=:])/[a-z]/[^[:space:]"'`)>]+
[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}
https?://[^[:space:])>"'`]+
[^[:space:]"'`()<>]*([Cc]onfluence|[Ss]harepoint|[Aa]tlassian|[Jj]ira|teams\.microsoft|slack\.com|wiki\.)[^[:space:]"'`()<>]*
(^|[^A-Za-z0-9/])[A-Z][A-Z0-9]+-[0-9]{2,}
[^[:space:]"'`()<>]*(TODO|FIXME|XXX|[Hh]andoff|[Kk]ickoff|[Ww]orktree|CLAUDE\.md|AGENTS\.md|\.cursor)[^[:space:]"'`()<>]*
```

Build the two term files. **Your own identifiers** are the likeliest leaks when the reviewer is the author. **Private terms**
(`$TERMS`: code names, cluster, site and node names, internal hosts) are what the repository cannot reveal; when absent, say so in
Coverage. Strip CR, surrounding whitespace and blank lines first. Whenever a block prints `STOP:`, the lane returns
`STOP: <reason>` instead of continuing.

```bash
. ./env.sh; L="$SCRATCH/LEAK"; . "$L/lib.sh"
printf '%s\n' "$HOME" "${USERPROFILE:-}" "${USER:-}" "${USERNAME:-}" "${COMPUTERNAME:-}" "$(hostname)" "$(hostname -f 2>/dev/null)" "$(git config user.email)" "$(git config user.name)" \
  "$CLONE" "$REPO" "$REPO_BASE" "$EVIDENCE" "$SCRATCH" | awk 'length >= 3 && !/^[[:space:]]*$/' | sort -u > "$L/local-terms.txt"
[ "$TERMS" = absent ] && TERMS=; if [ -n "$TERMS" ]; then [ -r "$TERMS" ] || { echo "STOP: terms file unreadable: $TERMS" >&2; exit 1; }
  tr -d '\r' < "$TERMS" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | awk 'NF' | sort -u; fi > "$L/terms.txt"

cat >> "$L/lib.sh" <<'EOF'
# search <-E|-F> <file>: skipped when empty (an empty -f file matches every line); -a reads past NUL bytes;
# patterns (-E) are case-sensitive, terms (-F) are not. The awk reads "<line>:<token>" and prints anchor, base path, token.
search() {
  [ -s "$2" ] || return 0
  local ci=i; [ "$1" = -E ] && ci=
  "$GREP" -noa$ci "$1" -f "$2" "$L/pub.content" | awk -v ANCH="$L/pub.anchor" '
    BEGIN { FS = ":"; while ((getline l < ANCH) > 0) { i++; split(l, p, "\t"); a[i] = p[1]; b[i] = p[2] } }
    { n = $1; t = substr($0, length(n) + 2)
      gsub("^[^[:alnum:]~$/-]+", "", t); gsub("[^[:alnum:]_/]+$", "", t)
      if (t != "") print a[n] "\t" b[n] "\t" t }'
}
scan() { { search -E "$L/patterns.txt"; search -F "$L/local-terms.txt"; search -F "$L/terms.txt"; } | sort -u; }
EOF
. "$L/lib.sh"
scan > "$L/hits.tsv"; wc -l < "$L/hits.tsv"; head -n 200 "$L/hits.tsv"
```

`hits.tsv` rows are `<anchor><TAB><base path><TAB><token>`; page with `sed -n`. Then check what patterns cannot match: committed
build output, virtual environments, caches, agent or planning notes, unpublished benchmark numbers, customer or partner names,
screenshots, and binaries (`git diff --numstat "$BASE" "$HEAD"` rows with `-` counts; `--summary` for mode changes and symlinks):
view images, list archives, list printable runs of code objects (`strings`, else `python -c` over the file's bytes).

## 3. Classify each hit

- **Already in `$BASE`?** The loop below checks the token in the hit's file. Present: not a PR finding; if identifying, list it
  under FUTURE WORK with its BASE location.
- **Identifying?** Names a real person, machine, host, path, credential or private system. Generic (`/home/user`,
  `user@example.com`): checked, no finding.
- **`TODO`/`FIXME`/`XXX` comment?** A leak only when it names a person, ticket, internal system or process artifact; otherwise
  checked, no finding (`CORR` owns code-quality TODOs).
- **Public?** Present at `$BASE`, or in a public ROCm repository or AMD's public docs (cite the URL). If you cannot show it is
  public: **Needs human decision**, never passed silently.

Hits in commit messages, identities, the branch name and PR text have no file: always `added`, as is a value present only elsewhere
in the tree (`jira` at `$BASE` does not clear a new `jira.<host>/...`).

```bash
. ./env.sh; L="$SCRATCH/LEAK"
while IFS=$'\t' read -r anchor bpath tok; do
  s=added
  if [ "$bpath" != - ]; then
    git -C "$CLONE" grep -qIiF -e "$tok" "$BASE" -- ":(literal)$bpath"
    case $? in 0) s=base ;; 1) ;; *) s=unknown ;; esac   # unknown: git failed; treat as added
  fi
  printf '%s\t%s\t%s\n' "$s" "$anchor" "$tok"
done < "$L/hits.tsv" > "$L/classified.tsv"
"$GREP" -cE '^(added|unknown)' "$L/classified.tsv"; "$GREP" -E '^(added|unknown)' "$L/classified.tsv" | head -n 200
```

A hit in a renamed file is checked under its old path. The author's commit identity and a `users/<name>/` branch prefix are expected
unless they differ from what the project publishes (`git -C "$CLONE" log --format=%ae -300 "$BASE"`). Tier:
[../policy.md](../policy.md) *Review rules* (leak tiers). Cite `path:line` at `$HEAD` (`<sha>:<path>:<line>` or `commit <sha>` when
not in the final diff). The lane report may quote a private term; text drafted for posting cites `path:line` only.

## Sweeping review output

For the output sweep, search the round's `REPORT.md`, `COMMENTS.md`, `CONTEXT.md`, every `.md` under `orient/` and `verify/`
(recursively, except `*-evidence/`), the published copy `$KNOWLEDGE/$KEY-r<N>/*.md` and `LEDGER.md` instead of the
published change: load them with `meta` (a missing file or `$KNOWLEDGE` copy fails the load), then run step 2 and filter.
Evidence subdirectories (`verify/<ID>-evidence/`, `lanes/*-evidence/`, `lanes/<ID>/`) are local-only and never shared, so
the sweep skips them; they may hold local paths and private terms:

```bash
. ./env.sh; L="$SCRATCH/LEAK"; . "$L/lib.sh"
load_output() {
  local RN; RN=$(basename "$PWD")   # the round directory, r<N>
  local R="$EVIDENCE/$RN" K="$KNOWLEDGE/$KEY-$RN" f n=0 miss=0
  for f in "$R/REPORT.md" "$R/COMMENTS.md" "$R/CONTEXT.md" "$EVIDENCE/LEDGER.md"; do
    if [ -f "$f" ]; then meta "${f#"$EVIDENCE"/}" < "$f"; n=$((n + 1))
    else echo "sweep: missing $f" >&2; miss=1; fi
  done
  while IFS= read -r f; do meta "${f#"$EVIDENCE"/}" < "$f"; n=$((n + 1))
  done < <(find "$R/orient" "$R/verify" -type f -name '*.md' -not -path '*-evidence/*' 2>/dev/null | sort)
  if [ -d "$K" ] && [ -n "$(find "$K" -type f -name '*.md' -print -quit)" ]; then
    while IFS= read -r f; do meta "knowledge/${f#"$KNOWLEDGE"/}" < "$f"; n=$((n + 1))
    done < <(find "$K" -type f -name '*.md' | sort)
  else echo "sweep: missing published copy $K" >&2; miss=1; fi
  echo "sweep: read $n files"; return $miss
}
: > "$L/pub.anchor"; : > "$L/pub.content"
load_output || { echo "STOP: sweep input missing, not clean" >&2; exit 1; }
sed -i.bak 's#users/[A-Za-z0-9._-]*/#users/#g' "$L/pub.content" && rm "$L/pub.content.bak"   # the expected branch prefix
cat > "$L/sweep-allow.txt" <<'EOF2'
^(CORR|TESTQ|TESTB|AUTH|BUILD|DEPS|API|ROCK|LEAK|DUP|DOCS|RUN|DRIFT|ORCH|VER|SCOUT|SYNTH)[0-9]*-p?[0-9]+$
^(handoffs?|worktrees?)$
^https?://(github\.com/ROCm/[A-Za-z0-9._-]+|raw\.githubusercontent\.com/ROCm/[A-Za-z0-9._-]+|rocm\.docs\.amd\.com|docs\.amd\.com|www\.amd\.com|example\.com)([/:?#].*)?$
EOF2
scan |
while IFS=$'\t' read -r anchor bpath tok; do
  "$GREP" -qiF -- "$tok" pr/pr.json 2>/dev/null && continue   # published by the author
  printf '%s\n' "$tok" | "$GREP" -qiE -f "$L/sweep-allow.txt" || printf '%s\t%s\n' "$anchor" "$tok"
done > "$L/sweep-hits.tsv"
wc -l < "$L/sweep-hits.tsv"; head -n 200 "$L/sweep-hits.tsv"
```

A hit counts unless `sweep-allow.txt` allows it or its token is in `pr/pr.json` (the author published it; the
source leak is a separate finding). A public URL outside the list is checked under `Public?`; a GitHub URL outside `ROCm/` is
public only when `gh api repos/<owner>/<repo> --jq .private` prints `false`. Write each counted hit to
`$EVIDENCE/r<N>/lanes/LEAK-sweep.md` as `<file>:<line>` without the token (see *Sweep dispatch*); the orchestrator redacts it in the file and in the `$KNOWLEDGE` copy to a repo-relative `path:line` or a
`$REPO`-style variable, then dispatches the sweep again. The sweep is clean when the load succeeded and `sweep-hits.tsv` is empty.
Write `Sweep: clean` only from a run of this block, after the last edit of any file it reads: `comments.py check-sweep` rejects a sweep older than any of them.
