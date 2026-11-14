#!/bin/bash
# AICK-3062 gfx1151 kit: start (or resume) the whole run inside a detached tmux session,
# so closing the terminal / SSH loses nothing. Safe to run again: it never starts a
# second copy, it just tells you how to look at the running one.
#
#   ./start.sh               setup (first time) + run_all.sh in tmux session "aick3062"
#   ./start.sh --autostart   same, and resume automatically after a reboot / power loss
#                            (adds an @reboot line to your crontab; remove with -e)
#   ./start.sh --status      show progress without attaching
#
# Watch:  tmux attach -t aick3062   (detach again: Ctrl-b then d)
set -u
KIT=$(cd "$(dirname "$0")" && pwd)
export AICK_WORK=${AICK_WORK:-$HOME/aick3062_gfx1151_work}
S=aick3062
command -v tmux >/dev/null || { echo "tmux not found: sudo apt install tmux"; exit 1; }
mkdir -p "$AICK_WORK"

if [ "${1:-}" = "--status" ]; then
  tmux has-session -t $S 2>/dev/null && echo "session $S: RUNNING" || echo "session $S: not running"
  tail -n 15 "$AICK_WORK/progress.txt" 2>/dev/null; ls -1t "$AICK_WORK"/checkpoints/ckpt-*.tgz 2>/dev/null | head -1
  exit 0
fi
if [ "${1:-}" = "--autostart" ]; then
  line="@reboot sleep 120 && AICK_WORK=$AICK_WORK $KIT/start.sh >> $AICK_WORK/autostart.log 2>&1"
  ( crontab -l 2>/dev/null | grep -v 'aick3062_gfx1151_kit/start.sh'; echo "$line" ) | crontab -
  echo "autostart on reboot: enabled (crontab -e to remove)"
fi
if grep -q '^DONE' "$AICK_WORK/progress.txt" 2>/dev/null; then
  echo "run already finished: send $AICK_WORK/gfx1151_results.tgz"; exit 0
fi
if tmux has-session -t $S 2>/dev/null; then
  echo "already running in tmux session '$S'. Watch: tmux attach -t $S   Status: $0 --status"
  exit 0
fi
CMD="cd '$KIT/..'; export AICK_WORK='$AICK_WORK'"
CMD="$CMD; [ -x \"\$AICK_WORK/venv/bin/python\" ] || '$KIT/setup.sh' 2>&1 | tee -a \"\$AICK_WORK/setup.log\""
CMD="$CMD; '$KIT/run_all.sh' 2>&1 | tee -a \"\$AICK_WORK/run.log\""
CMD="$CMD; echo; echo 'run_all.sh exited - see progress.txt. This window stays open.'; exec bash"
tmux new-session -d -s $S "bash -c \"$CMD\""
echo "started in tmux session '$S' (work dir: $AICK_WORK)"
echo "  watch:  tmux attach -t $S      (detach: Ctrl-b then d)"
echo "  status: $0 --status"
