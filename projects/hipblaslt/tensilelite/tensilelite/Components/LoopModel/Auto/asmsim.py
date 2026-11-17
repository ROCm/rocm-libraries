# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tier 3: time the steady loop StinkyTofu actually emitted for a candidate.

Tiers 0-2 price a candidate from its parameters; this one emits the kernel (CPU-only) and runs
the body between `label_LoopBeginL:` and its back edge through an in-order issue model, so it
sees the schedule, the waitcnts and the barriers the scheduler really produced:

  v_wmma*           holds the matrix queue `wmma` cycles; everything else co-issues
  ds_load*          returns `asmDsLatency` after issue, no sooner than `dsCyclesPerLoad` after
                    the previous return
  s_wait_dscnt N    waits until at most N ds_loads are outstanding
  tensor_load*      lands `asmTdmLatency` after issue; s_wait_tensorcnt N as above
  s_barrier_wait    costs `tripOverhead`'s barrier share
  anything else     one issue cycle

The body is run several times back to back and the last copy timed, so loads a copy prefetches
for the next are charged where they are waited on.
"""

import re

_BACK_EDGE = re.compile(r"\s*s_cbranch_\w+\s+label_LoopBeginL\b")
_BARRIER_CYCLES = 11


def loopBody(source):
  """The steady-loop instructions of an emitted kernel, or None when it has no LoopBeginL."""
  lines = source.splitlines()
  try:
    start = next(i for i, line in enumerate(lines) if line.startswith("label_LoopBeginL:"))
  except StopIteration:
    return None
  ends = [i for i, line in enumerate(lines) if _BACK_EDGE.match(line)]
  if not ends:
    return None
  body = []
  for line in lines[start + 1:ends[-1] + 1]:
    line = line.split("//")[0].strip()
    if line and not line.endswith(":"):
      body.append(line)
  return body


def _count(inst):
  return int(inst.split()[1].rstrip(","), 0)


def loopCycles(body, costs, wmma, wmmasPerTrip, rounds=3):
  """Cycles per trip of `body`, a `loopBody`, under `costs`."""
  clock = wmmaFree = lastDs = 0.0
  ds, tdm = [], []
  start = 0.0
  for round_ in range(rounds):
    if round_ == rounds - 1:
      start = max(clock, wmmaFree)
    for inst in body:
      op = inst.split(None, 1)[0]
      if op.startswith("v_wmma"):
        wmmaFree = max(clock, wmmaFree) + wmma
      elif op.startswith("ds_load") or op.startswith("ds_read"):
        lastDs = max(clock + costs.asmDsLatency, lastDs + costs.dsCyclesPerLoad)
        ds.append(lastDs)
      elif op == "s_wait_dscnt":
        keep = _count(inst)
        while len(ds) > keep:
          clock = max(clock, ds.pop(0))
      elif op.startswith("tensor_load"):
        tdm.append(clock + costs.asmTdmLatency)
      elif op == "s_wait_tensorcnt":
        keep = _count(inst)
        while len(tdm) > keep:
          clock = max(clock, tdm.pop(0))
      elif op == "s_barrier_wait":
        clock += _BARRIER_CYCLES - 1
      clock += 1
  wmmas = sum(1 for inst in body if inst.startswith("v_wmma"))
  trips = max(1, wmmas // max(1, wmmasPerTrip))
  return (max(clock, wmmaFree) - start) / trips
