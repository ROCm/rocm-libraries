#!/usr/bin/env bash
# rt_cpu_submit.sh <version>: stage the rust-track tree, build + gate the crate on CPUONLY&MARKHAM (rt_cpu.sh), wait,
# then publish the .so and gate binaries as the q38 stage rt-bin-<version> (Austin home) for GPU jobs.
#   stage names: rocke-rt-<version> (the tree), rt-bin-<version> (so/ + bin/); results: q38-rt/out/rt-cpu-<version>/
set -uo pipefail
V=${1:?version}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/c/Users/vanantha/Documents/dev/hackathon-prep/tools
L=vanantha@ctr2-alola-login-01
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30"
TREE=/c/Users/vanantha/Documents/dev/copy-rocm-lib-rust
ST=rocke-rt-$V
TAG=rt-cpu-$V
AUS=/home_aus/AMD/vanantha/rocke-sdpa-claude/q38
MK=/home/AMD/vanantha/rocke-sdpa-claude/q38-rt
crlf=$(tr -d -c $'\r' < "$HERE/rt_cpu.sh" | wc -c); [ "$crlf" = 0 ] || { echo "rt_cpu.sh has CRLF"; exit 2; }
bash "$T/q38/q38_sync.sh" "$TREE" "$ST" | tail -1 || exit 1
$SSH $L "mkdir -p $MK/stages $MK/jobs $MK/out $MK/logs && cp $AUS/stages/$ST.tar $MK/stages/ && cp $AUS/stages/$ST.tar.sha256 $MK/stages/ && cat > $MK/jobs/rt_cpu.$V.sh && chmod +x $MK/jobs/rt_cpu.$V.sh" < "$HERE/rt_cpu.sh" || exit 1
jid=$($SSH $L "cd /tmp && sbatch --parsable -p defq -C 'CPUONLY&MARKHAM' --exclude=ctr-smc-s22-025 -c 32 --mem=64G -t 00:40:00 -J rt-cpu-$V -o $MK/logs/$TAG-%j.out --export=ALL,TAG=$TAG,RT_STAGE=$ST $MK/jobs/rt_cpu.$V.sh") || exit 1
echo "[rt] submitted $jid ($TAG)"
for i in $(seq 1 240); do
  sleep 30
  if $SSH $L "grep -q '^== rc=' $MK/logs/$TAG-$jid.out 2>/dev/null"; then break; fi
done
$SSH $L "cat $MK/logs/$TAG-$jid.out" | grep -E "^== " | cut -c1-300
rc=$($SSH $L "grep -o '^== rc=[0-9]*' $MK/logs/$TAG-$jid.out | tail -1" | sed 's/^== rc=//')
echo "[rt] $TAG rc=${rc:-none}"
# publish so/ + bin/ as an Austin stage even on a gate failure (the GPU jobs check the gates' rc themselves)
$SSH $L "cd $MK/out/$TAG && tar -cf $AUS/stages/.rt-bin-$V.tar so bin logs census_compare.txt && mv $AUS/stages/.rt-bin-$V.tar $AUS/stages/rt-bin-$V.tar && sha256sum $AUS/stages/rt-bin-$V.tar | cut -d' ' -f1 > $AUS/stages/rt-bin-$V.tar.sha256 && echo '[rt] stage rt-bin-$V' && cat so/librocke_codeobj_ffi.so.sha256"
mkdir -p /c/Users/vanantha/Documents/dev/hackathon-prep/logs/q38/rt
$SSH $L "cd $MK/out/$TAG && tar -cf - logs census_compare.txt scope_compare.txt" | tar -xf - -C /c/Users/vanantha/Documents/dev/hackathon-prep/logs/q38/rt/ --one-top-level=$TAG 2>/dev/null
exit "${rc:-124}"
