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

# 2. one-time setup (~6 GB torch download); work dir defaults to ~/aick3062_gfx1151_work
./aick3062_gfx1151_kit/setup.sh

# 3. run everything, detached from the terminal
nohup ./aick3062_gfx1151_kit/run_all.sh > ~/aick3062_gfx1151_run.log 2>&1 &

# 4. follow progress
tail -f ~/aick3062_gfx1151_work/progress.txt
```
`run_all.sh` can be resumed: if the machine reboots or the run stops, start step 3 again
with the same command and it continues where it stopped.

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

The work dir also holds `gate_constants_gfx1151.json`. Keep it private: don't commit it or
post it publicly.

## If something goes wrong
- `PROBE_FAIL no gfx1151`: `rocminfo` doesn't see a gfx1151 GPU (driver or groups).
- `STOP sweep ...` / `STOP time ...`: see the matching `*.log` in the work dir, and send
  that log together with `progress.txt`.
- A different work dir: set `AICK_WORK=/path` for both `setup.sh` and `run_all.sh`.
