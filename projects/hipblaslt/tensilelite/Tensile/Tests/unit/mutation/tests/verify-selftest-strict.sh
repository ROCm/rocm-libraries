#!/usr/bin/env bash
# verify-selftest-strict.sh — docker-free unit check of mutmut-verify.sh STRICT
# kill classification (Issue 1). Sources ONLY the pure classify_verdict() helper
# via MUTMUT_VERIFY_LIB_ONLY (no docker, no mutmut, no pytest), then asserts the
# decision truth table: KILLED only when the mutant node fails with rc==1.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
MUTMUT_VERIFY_LIB_ONLY=1 source "$HERE/../mutmut-verify.sh"

fail=0
v() { classify_verdict "$@" | cut -f1; }   # verdict field only
check() { # desc expected actual
  if [[ "$3" == "$2" ]]; then echo "ok   - $1 ($3)"; else echo "FAIL - $1: expected $2 got $3"; fail=1; fi
}

#     desc                                         expected     base exp mut want_fail revert
check "assertion failure rc1 => KILLED"            KILLED       "$(v 0 0 1 true ok)"
check "mutant passes rc0 => survived (not killed)" BAD          "$(v 0 0 0 true ok)"
check "collection error rc2 => INCONCLUSIVE"       INCONCLUSIVE "$(v 0 0 2 true ok)"
check "usage error rc3 => INCONCLUSIVE"            INCONCLUSIVE "$(v 0 0 3 true ok)"
check "internal error rc4 => INCONCLUSIVE"         INCONCLUSIVE "$(v 0 0 4 true ok)"
check "interrupt rc5 => INCONCLUSIVE"              INCONCLUSIVE "$(v 0 0 5 true ok)"
check "baseline mismatch => BAD"                   BAD          "$(v 1 0 1 true ok)"
check "revert leak => BAD"                         BAD          "$(v 0 0 1 true LEAK)"
check "want_fail=false + rc0 => OK (not a kill)"   OK           "$(v 0 0 0 false ok)"
check "want_fail=false + rc1 => BAD"               BAD          "$(v 0 0 1 false ok)"
check "non-numeric exp field => BAD (no abort)"    BAD          "$(v 0 abc 1 true ok)"

# Prove the old logic is gone: rc!=0 alone must NOT yield KILLED for rc 2..5.
for rc in 2 3 4 5; do
  [[ "$(v 0 0 "$rc" true ok)" == "KILLED" ]] && { echo "FAIL - rc=$rc wrongly KILLED (old mut_rc!=0 logic)"; fail=1; }
done
# KILLED must be emitted ONLY for mut_rc==1: no other (rc, mode) combination may KILL.
for mode in true false; do for rc in 0 2 3 4 5 137; do
  [[ "$(v 0 0 "$rc" "$mode" ok)" == "KILLED" ]] && { echo "FAIL - rc=$rc mode=$mode wrongly KILLED (KILLED must require mut_rc==1)"; fail=1; }
done; done

if [[ $fail -eq 0 ]]; then echo "ALL SELFTESTS PASSED"; else echo "SELFTEST FAILURES"; exit 1; fi
