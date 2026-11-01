#!/usr/bin/env bash
# rt_mut_submit.sh <version>: build the GPU mutant libraries of rt_mutants.py from the staged tree rocke-rt-<version>
# (already in the Markham home from rt_cpu_submit.sh) and publish them as the q38 stage rt-mut-<version>.
set -uo pipefail
V=${1:?version}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=vanantha@ctr2-alola-login-01
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30"
AUS=/home_aus/AMD/vanantha/rocke-sdpa-claude/q38
MK=/home/AMD/vanantha/rocke-sdpa-claude/q38-rt
TAG=rt-mut-$V
$SSH $L "cat > $MK/jobs/rt_mutants.py" < "$HERE/../rt_mutants.py" || exit 1
$SSH $L "cat > $MK/jobs/rt_mut.$V.sh && chmod +x $MK/jobs/rt_mut.$V.sh" < "$HERE/rt_mut.sh" || exit 1
jid=$($SSH $L "cd /tmp && sbatch --parsable -p defq -C 'CPUONLY&MARKHAM' --exclude=ctr-smc-s22-025 -c 32 --mem=64G -t 00:40:00 -J rt-mut-$V -o $MK/logs/$TAG-%j.out --export=ALL,TAG=$TAG,RT_STAGE=rocke-rt-$V $MK/jobs/rt_mut.$V.sh") || exit 1
echo "[rt] submitted $jid ($TAG)"
for i in $(seq 1 240); do
  sleep 30
  if $SSH $L "grep -q '^== rc=' $MK/logs/$TAG-$jid.out 2>/dev/null"; then break; fi
done
$SSH $L "cat $MK/logs/$TAG-$jid.out" | grep -E "built|FAILED|^== " | cut -c1-200
$SSH $L "cd $MK/out/$TAG && tar -cf $AUS/stages/.rt-mut-$V.tar mut_*.so && mv $AUS/stages/.rt-mut-$V.tar $AUS/stages/rt-mut-$V.tar && sha256sum $AUS/stages/rt-mut-$V.tar | cut -d' ' -f1 > $AUS/stages/rt-mut-$V.tar.sha256 && echo '[rt] stage rt-mut-$V'"
