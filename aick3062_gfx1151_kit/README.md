# AICK-3062: gfx1151 tuned-table run kit (rocKE SDPA backward)

A self-contained kit to produce the **gfx1151 (Strix Halo)** tuned-table entries for the
rocKE SDPA backward on a local machine. It goes with PR ROCm/rocm-libraries#15795.
Every measurement is made on your machine: the kit carries code and scripts only, no data.

This folder is a run kit, not product code: it is never merged.

## What you need
- A **gfx1151** GPU (Radeon 8060S / 8050S class) with a working amdgpu driver.
  `rocminfo | grep gfx1151` must print a line.
- `rocminfo` installed (the ROCm `rocminfo` package). A full ROCm install is not needed:
  the torch wheel brings its own ROCm 7.2 runtime and compiler.
- Your user in the `render` and `video` groups.
- Python 3.12 (with `venv`), `git`, about 15 GB free disk.
- The machine to itself for about **1-2 days**, on AC power, with no other GPU work.
  The timings become the table, so background load matters.

## Steps
```bash
# 1. sparse clone of just this kit and rocKE (~75 MB of files)
git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout set dnn-providers/hip-kernel-provider/rocke aick3062_gfx1151_kit
git checkout users/tlakshma/rocke/aick3062-gfx1151-kit

# 2. start: setup (first time, ~6 GB torch download) + the whole run, in tmux
./aick3062_gfx1151_kit/start.sh --autostart     # --autostart: resume after a reboot

# 3. check on it any time (or: tmux attach -t aick3062, then Ctrl-b d to detach)
./aick3062_gfx1151_kit/start.sh --status
```
`tmux` is required (`sudo apt install tmux`). Work dir: `~/aick3062_gfx1151_work`.
Running `start.sh` again is always safe: it resumes a stopped run and never starts a
second copy.

## If something stops abruptly
Nothing measured is lost, at three levels:
1. **Resume:** every step records each finished shape and stage as it goes, and a
   restart skips work already recorded. Just run `./aick3062_gfx1151_kit/start.sh` again
   (with `--autostart` this also happens by itself after a reboot).
2. **Repair:** on every start, a record half-written at the moment of a crash is cut off
   cleanly, so at most the one shape in flight is re-timed.
3. **Checkpoints:** all results are snapshotted every 20 minutes and after every step to
   `~/aick3062_gfx1151_work/checkpoints/` (the last 6 are kept). Set
   `CHECKPOINT_DIR=/path/on/another/disk` before `start.sh` to also copy each snapshot
   there. To recover a damaged work dir:
   ```bash
   cd ~/aick3062_gfx1151_work
   tar xzf "$(ls -1t checkpoints/ckpt-*.tgz | head -1)"
   ~/rocm-libraries/aick3062_gfx1151_kit/start.sh       # adjust to your clone path
   ```

## What it does
1. **Sweep** (~5-6 h): the general-family funnel sweep, stages `geometry` to `runtime`,
   head sizes 64 and 128. It keeps 2 finalist configurations per head size.
2. **Calibrate** (seconds): the bench's gate thresholds, from this device's own noise.
3. **Time** (most of a day): rocKE's shipped plan, the finalists (each with an A/A twin)
   and PyTorch's default backward, on every dense census shape (23 at d64, 55 at d128).
4. **Judge** (seconds): applies the RDNA entry rule, documented in
   `dispatch/attention_bwd/gfx1151_tuning.py`. An entry is a finalist that, on every shape
   of its key, is correct, has a clean A/A check, and is not slower than the shipped plan.
5. **Pack:** writes `~/aick3062_gfx1151_work/gfx1151_results.tgz`.

## When it finishes
`progress.txt` ends with `DONE ... send .../gfx1151_results.tgz`.
**Send `gfx1151_results.tgz` to Thrupti (thruptiraj.lakshmanagowda@amd.com)** by email,
Teams or SharePoint, *not* through GitHub. It holds raw timing data, which the project
keeps out of the repository.

The work dir (and its checkpoints) also holds `gate_constants_gfx1151.json`. Keep it
private: don't commit it or post it publicly.

## If something goes wrong
- `PROBE_FAIL no gfx1151`: `rocminfo` doesn't see a gfx1151 GPU (driver or groups).
- `STOP sweep ...` / `STOP time ...`: see the matching `*.log` in the work dir, and send
  that log together with `progress.txt`.
- A different work dir: set `AICK_WORK=/path` before `start.sh`.
- The tmux window stays open after the run ends; `tmux kill-session -t aick3062` closes it.
