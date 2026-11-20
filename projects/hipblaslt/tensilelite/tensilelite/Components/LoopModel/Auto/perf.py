# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a LoopModel candidate costs per main-loop trip, in cycles of one wave.

Tier 1, `estimate`, prices θ analytically and is cheap enough for every candidate:

  T_trip = max(C_wmma, C_lds) + sum over operands of the exposed ds_load latency + overhead

  C_wmma    WMMAs x the cycles each holds the matrix queue (WmmaQueueTimeline)
  C_lds     ds_load instructions x the LDS return rate
  exposed   max(0, L_ds - D) per refill.  D is the ds_load->wmma DISTANCE: the read-ahead steps
            the ring can actually run (PLR, less any step the ring has no free slot for) times
            the WMMAs one step spans.  L_ds is the burst's drain latency (StinkyTofu's
            `computeDynamicDrainLatency`) plus the queueing the other loads add at the LDS
            utilization C_lds / T.

Tier 2, `simulate`, walks the lowered GIR steady body -- the real ds_load placement -- through
an in-order issue clock with a WMMA queue, the LDS return queue and the RAW/WAR register
dependences.  StinkyTofu's scheduler still reorders after GIR, so this prices the GIR intent.

Both tiers charge each tensor_load_to_lds its issue (`tdmIssueCycles`, `tensorLoads` per trip);
tier 2 also makes a ds_load of an LDS generation wait for the tensor_load filling it, and that
load wait for the last read of the buffer's previous generation.  The s_wait_tensorcnt depth
StinkyTofu emits is set on the scheduled loop, so neither tier sees it; tier 3 does.
"""

from dataclasses import dataclass, field
from math import ceil

from ....LoopModel import traversal as _geometry


@dataclass
class PerfEstimate:
  cycles: float                  #: per trip, per wave
  bound: str                     #: wmma, lds or latency
  wmma: float
  lds: float
  exposed: dict = field(default_factory=dict)    #: {operand: cycles per trip}
  distance: dict = field(default_factory=dict)   #: {operand: ds_load->wmma cycles}
  latency: dict = field(default_factory=dict)    #: {operand: effective ds latency}
  dsLoads: dict = field(default_factory=dict)    #: {operand: ds_load instructions per trip}
  codeSize: int = 0
  extra: dict = field(default_factory=dict)

  def summary(self):
    return ("%.0f cyc/trip (%s-bound: wmma %.0f, lds %.0f, exposed %s)"
            % (self.cycles, self.bound, self.wmma, self.lds,
               {k: round(v) for k, v in self.exposed.items() if v}))


def _mx(kernel):
  pt = kernel["ProblemType"]
  return bool(pt.get("MXBlockA") or pt.get("MXBlockB"))


def wmmaCycles(kernel, costs):
  mi = kernel["MatrixInstruction"]
  return costs.wmma(mi[0], mi[1], mi[2], _mx(kernel))


def _readBytesPerLane(kernel, operand):
  """Bytes one ds_load moves per lane for `operand`."""
  side = "A" if operand.name.endswith("A") else "B"
  if operand.name.startswith("MXS"):
    return 4
  pt = kernel["ProblemType"]
  dataType = pt.get("DataType" + side) or pt["DataType"]
  lrvw = int(kernel.get("LocalReadVectorWidth" + side, 0) or 0)
  if lrvw <= 0:
    lrvw = int(kernel.get("LocalReadVectorWidth", 0) or 0)
  return max(4, min(16, int(lrvw * dataType.numBytes()))) if lrvw > 0 else 16


def tensorLoads(kernel):
  """tensor_load_to_lds instructions per trip: one per storage region of each descriptor set
  (a fused set -- A with its MX scales -- is one walk)."""
  from ...TDMFuse import tdmFusedGroups
  from ...TDMSplit import set_regions
  operands = [tc for tc in ("A", "B", "MXSA", "MXSB") if kernel.get("enableTDM" + tc)]
  sets, seen = [], set()
  for tc in operands:
    if tc in seen:
      continue
    members = next((g for g in tdmFusedGroups(kernel) if tc in g), (tc,))
    seen.update(members)
    sets.append(set_regions(kernel, tc))
  return sum(sets)


def tdmExposed(cWmma, cLds, copies, costs):
  """tensor_load_to_lds latency a trip leaves exposed, from its steady-body copies.

  The wait depth is set after scheduling (StinkyWaitCntInsertionPass): a one-copy body waits
  for s_wait_tensorcnt 0 every trip, a fraction of a trip after the load it retires issued, so
  nearly all the latency shows; a longer body keeps each load in flight about one trip.
  """
  return max(0.0, costs.tdmLatency - (max(cWmma, cLds) if copies > 1 else 0.0))


def drainLatency(costs, latency, loads, waves):
  """StinkyTofu `computeDynamicDrainLatency`: one burst of `loads` ds_loads, `waves` resident."""
  depth = costs.dsQueueDepth
  if depth <= 0 or loads <= 1:
    return latency
  if loads <= depth:
    return latency + (loads - 1) * waves
  return latency + (depth - 1) * waves + (loads - depth) * waves / 2


def _readOperands(theta):
  return [operand for operand in theta.operands
          if not operand.is_output and operand.trajectory.fragment_fill is not None]


def estimate(kernel, theta, S, costs, target=None):
  """Tier 1: the analytic per-trip cost of the candidate θ describes."""
  shape = _geometry.mma_shape(theta)
  wmmas = int(shape.m_tiles) * int(shape.n_tiles) * int(shape.k_tiles)
  perWmma = wmmaCycles(kernel, costs)
  cWmma = wmmas * perWmma
  waves = max(1, int(kernel.get("NumThreads", 128)) // int(kernel.get("WavefrontSize", 32)))
  instances = ((target or {}).get("ReadInstances") or {})

  loads, units, burst, latency = {}, {}, {}, {}
  for operand in _readOperands(theta):
    tiles = _geometry.presence_tiles(theta, operand)
    merge = max(1, int(instances.get(operand.name, 1) or 1))
    moves = max(1, tiles // merge)
    regsPerMove = max(1, _geometry.frag_regs(theta, operand) * (merge if operand.name.startswith("MXS") else 1))
    bytesPerLane = _readBytesPerLane(kernel, operand)
    perMove = 1 if operand.name.startswith("MXS") else max(1, ceil(regsPerMove * 4 / bytesPerLane))
    loads[operand.name] = moves * perMove
    pu = max(1, _geometry.prefetch_unit_tiles(theta, operand) // max(1, _geometry.vgpr_group(theta, operand)))
    units[operand.name] = max(1, tiles // pu)
    burst[operand.name] = max(1, loads[operand.name] // units[operand.name])
    latency[operand.name] = costs.dsLoadLatency(bytesPerLane)
  cLds = sum(loads.values()) * costs.dsCyclesPerLoad
  rho = min(0.95, cLds / float(max(cWmma, cLds, 1)))

  exposed, distance, effective = {}, {}, {}
  for operand in _readOperands(theta):
    name = operand.name
    parent = "A" if name.endswith("A") else "B"
    plr = int(kernel.get("PrefetchLocalRead" + parent, 0) or 0)
    va = int(kernel.get("VgprAlloc" + parent, 1) or 1)
    # A refill can only run ahead into a slot nobody still reads: VA buffers hold the unit in
    # use plus VA-1 read-ahead units.
    steps = max(0, min(plr, va - 1)) if va > 1 else 0
    perStep = wmmas / float(units[name])
    distance[name] = steps * perStep * perWmma
    service = burst[name] * costs.dsCyclesPerLoad
    queueing = rho / (2.0 * (1.0 - rho)) * service
    effective[name] = drainLatency(costs, latency[name], burst[name], waves) + queueing
    exposed[name] = max(0.0, effective[name] - distance[name]) * units[name] \
        if not name.startswith("MXS") else 0.0

  copies = _geometry.ring_trip_copies(theta, S)
  body = (wmmas + sum(loads.values())) * copies
  icache = max(0.0, body - costs.icacheBodyInstructions) / float(max(1, copies)) * 0.25
  stall = sum(exposed.values())
  core = max(cWmma, cLds)
  total = core + stall + costs.tripOverhead + icache
  bound = "latency" if stall > 0.25 * core else ("lds" if cLds > cWmma else "wmma")
  return PerfEstimate(total, bound, cWmma, cLds, exposed, distance, effective, loads, body,
                      {"rho": rho, "copies": copies, "icache": icache, "wmmas": wmmas})


# --- tier 0: the nest walk, priced before θ is built --------------------------------------------

class NestWalk:
  """Where one loop order first uses each prefetch unit, from the nest alone.

  The walk is the WMMA sequence of one trip -- every point of the ordered axes, outermost first.
  An operand's unit is one value of its axes outside the unit, plus which of the VG sub-units
  of the unit axes the point falls in; a ds_load must land before the FIRST WMMA that uses it,
  and PLR moves its issue that many first-uses earlier.  Revisiting a unit costs no load: θ
  counts one load per presence tile (`presence_tiles`), which every lowered candidate
  confirms.
  """

  def __init__(self, kernel, shapes):
    from .. import Order
    axes = [(name, int(extent)) for name, extent in Order.axes(kernel)["axes"]]
    self.axes = axes
    self.wmmas = 1
    for _name, extent in axes:
      self.wmmas *= max(1, extent)
    self.shapes = shapes
    self._firstUses = {}

  def firstUses(self, side, vg):
    """Sorted WMMA positions at which `side`'s units (VG sub-units) are first used."""
    return [first for first, _last in self.uses(side, vg)]

  def uses(self, side, vg):
    """`[(first, last)]` WMMA positions of each of `side`'s units (VG sub-units), by first use."""
    key = (side, vg)
    if key in self._firstUses:
      return self._firstUses[key]
    shape = self.shapes[side]
    roles = {"A": ("M", "K"), "B": ("N", "K")}[side]
    unitNames = [name for name, _extent in shape.unitModes]
    unitExtents = dict(shape.unitModes)
    perSub = max(1, shape.unit // max(1, vg))
    names = [name for name, _extent in self.axes]
    extents = [extent for _name, extent in self.axes]
    seen, uses = {}, []
    coord = [0] * len(names)
    for position in range(self.wmmas):
      rest = position
      for index in range(len(names) - 1, -1, -1):
        coord[index] = rest % max(1, extents[index])
        rest //= max(1, extents[index])
      flat, stride = 0, 1
      for name in reversed(unitNames):
        flat += coord[names.index(name)] * stride
        stride *= unitExtents[name]
      unitKey = tuple(coord[i] for i, name in enumerate(names)
                      if name[:1] in roles and name not in unitExtents) + (flat // perSub,)
      if unitKey not in seen:
        seen[unitKey] = len(uses)
        uses.append([position, position])
      else:
        uses[seen[unitKey]][1] = position
    uses = [tuple(use) for use in uses]
    self._firstUses[key] = uses
    return uses

  def distances(self, side, vg, steps, buffers=None):
    """WMMAs between each unit's load issue and its first use, or None when the ring is too
    small to be correct.

    A unit's load issues `steps` first-uses early, but no earlier than the last use of the unit
    that held its buffer before it: a CLR=0 ring of `buffers` slots hands them out in first-use
    order, and a unit the walk comes back to (an A unit across N) keeps its slot until then.
    A slot not free by the unit's own first use is the plan's WRONG SOURCE.  `buffers=None` is
    a resident (CLR=1) operand, whose units never share a slot.
    """
    stream = self._stream(side, vg, steps, buffers)
    if stream is None:
      return None
    n = len(stream) // 3
    return [first - issued for issued, first in stream[2 * n:]]

  def exposure(self, side, vg, steps, buffers, latency, perWmma):
    """Cycles per trip the matrix queue waits on `side`'s ds_loads, or None like `distances`.

    One in-order clock over three trips laid end to end, keeping the third: a unit's wait
    delays every later WMMA, and a load issued before that wait keeps draining through it, so a
    burst of refills that are all short of `latency` pays about one latency, not one each.
    """
    stream = self._stream(side, vg, steps, buffers)
    if stream is None:
      return None
    n = len(stream) // 3
    waits, total = [], 0.0
    for g, (issued, first) in enumerate(stream):
      before = sum(wait for at, wait in waits if at < issued)
      wait = max(0.0, issued * perWmma + before + latency - (first * perWmma + total))
      waits.append((first, wait))
      total += wait
    return sum(wait for _at, wait in waits[2 * n:])

  def _stream(self, side, vg, steps, buffers):
    """`[(issued, first)]` WMMA positions of the units of three consecutive trips."""
    uses = self.uses(side, vg)
    n = len(uses)
    if n == 0:
      return []

    def at(g):   # (first, last) of the g-th unit of the steady stream, trips laid end to end
      first, last = uses[g % n]
      shift = self.wmmas * (g // n)
      return first + shift, last + shift

    out = []
    for g in range(3 * n):
      first, _last = at(g)
      issued = at(g - steps)[0] if steps > 0 else first
      if buffers is not None:
        free = at(g - buffers)[1] + 1
        if free > first:
          return None
        issued = max(issued, free)
      out.append((issued, first))
    return out


@dataclass(frozen=True)
class SideCost:
  regs: int
  exposed: float
  copies: int
  steps: int
  units: int     #: units (VG sub-units) loaded per trip, each paying the latency once


def sideCost(kernel, walk, side, choice, costs, rho, mx):
  """Tier 0: one operand's VGPRs and exposed ds latency per trip under `choice`, or None when
  its ring is too small to be correct (`NestWalk.distances`)."""
  shape = walk.shapes[side]
  pt = kernel["ProblemType"]
  dataType = pt.get("DataType" + side) or pt["DataType"]
  fragRegs = max(1, int(kernel["MIInputPerThread" + side]) * dataType.numBytes() // 4)
  perBuffer = max(1, shape.unit // max(1, choice.vg))
  regs = perBuffer * choice.va * (fragRegs + (1 if mx else 0))
  bytesPerLane = max(4, min(16, int(kernel.get("LocalReadVectorWidth" + side, 4) or 4)
                         * dataType.numBytes()))
  perTile = max(1, ceil(fragRegs * 4 / bytesPerLane))
  # The unit's own drain is a floor; under the loop's load the effective latency dominates it.
  waves = max(1, int(kernel.get("NumThreads", 128)) // int(kernel.get("WavefrontSize", 32)))
  latency = max(costs.dsEffectiveLatency,
                drainLatency(costs, costs.dsLoadLatency(bytesPerLane), perBuffer * perTile,
                             waves))
  steps = choice.plr if choice.clr else max(0, min(choice.plr, choice.va - 1))
  perWmma = wmmaCycles(kernel, costs)
  buffers = None if choice.clr else choice.va
  distances = walk.distances(side, choice.vg, steps, buffers)
  if distances is None:
    return None
  exposed = walk.exposure(side, choice.vg, steps, buffers, latency, perWmma)
  from math import gcd
  copies = 1 if choice.clr else choice.va // gcd(choice.va, max(1, shape.positions))
  return SideCost(regs, exposed, copies, steps, len(distances))


def orderCore(kernel, walk, costs):
  """Tier 0: `(C_wmma, C_lds, rho, ds_loads)` for one loop order -- the same for every register
  choice, because each presence tile is loaded once."""
  pt = kernel["ProblemType"]
  mx = bool(pt.get("MXBlockA") or pt.get("MXBlockB"))
  cWmma = walk.wmmas * wmmaCycles(kernel, costs)
  loads = 0
  for side in ("A", "B"):
    shape = walk.shapes[side]
    dataType = pt.get("DataType" + side) or pt["DataType"]
    fragRegs = max(1, int(kernel["MIInputPerThread" + side]) * dataType.numBytes() // 4)
    bytesPerLane = max(4, min(16, int(kernel.get("LocalReadVectorWidth" + side, 4) or 4)
                           * dataType.numBytes()))
    loads += shape.residentTiles * max(1, ceil(fragRegs * 4 / bytesPerLane))
    if mx and pt.get("MXBlock" + side):
      loads += max(1, shape.residentTiles // 2)
  cLds = loads * costs.dsCyclesPerLoad
  rho = min(0.95, cLds / float(max(cWmma, cLds, 1)))
  return cWmma, cLds, rho, loads


# --- tier 2: the GIR steady-body timeline -------------------------------------------------------

@dataclass
class Timeline:
  cycles: float          #: steady-state cycles per trip
  wmmaBusy: float        #: matrix-queue busy cycles per trip
  rawWait: float         #: cycles per trip the matrix queue sat idle waiting on a ds_load
  warWait: float         #: cycles per trip refills waited on the WMMA still reading their slot
  dsQueueWait: float     #: cycles per trip issue waited on a full LDS return queue
  copies: int
  tdmWait: float = 0.0   #: cycles per trip issue waited on a tensor_load (landing, or its buffer)
  tdmIssue: float = 0.0  #: cycles per trip spent issuing tensor_loads

  def summary(self):
    return ("%.0f cyc/trip (wmma busy %.0f, raw %.0f, war %.0f, dsq %.0f, tdm %.0f+%.0f)"
            % (self.cycles, self.wmmaBusy, self.rawWait, self.warWait, self.dsQueueWait,
               self.tdmIssue, self.tdmWait))


def _tileKey(ref):
  return (ref.tile.operand, tuple(ref.tile.coord))


def _slotKey(ref):
  return (ref.tile.operand, ref.group, ref.slot)


def _bufferKey(ref, shift):
  """`(generation id, absolute generation)` of a shared ref; steady copies step `gdelta`."""
  if ref.tile.space != "shared" or ref.gen is None:
    return None
  return (ref.gen.id, int(ref.gdelta) + shift)


def simulate(kernel, program, costs, rounds=3, wmmaQueue=4):
  """Tier 2: steady-state cycles per trip of `program`'s steady body (one wave)."""
  from ....Lowering.gir.nodes import Mma, Move

  steady = [label for label in program.blocks if str(label).startswith("steady")]
  if not steady:
    raise ValueError("program has no steady block")
  perWmma = wmmaCycles(kernel, costs)
  waves = max(1, int(kernel.get("NumThreads", 128)) // int(kernel.get("WavefrontSize", 32)))
  body = [inst for _round in range(rounds) for label in steady
          for inst in program.block(label).body]
  trip = len(steady)
  marks = []          # issue clock at each trip start
  clock = 0.0
  pipeEnd = 0.0
  pipeEnds = []
  ldsFree = 0.0
  inflight = []       # completion times of outstanding ds_loads, oldest first
  ready = {}          # tile key -> ready time
  lastRead = {}       # slot key -> time the last WMMA reading it started
  # A tensor_load fills one LDS generation; a ds_load of it waits for the landing, and the next
  # fill of the same buffer (`ring` generations on) waits for the last ds_load of this one.  A
  # generation the simulated rounds never fill came from the prologue and is ready at 0.
  landed = {}         # buffer key -> time its tensor_load lands in LDS
  drained = {}        # buffer key -> time the last ds_load reading it returned
  raw = war = dsq = busy = tdm = tdmIssue = 0.0
  counters = []
  blockStarts = set()
  index = 0
  for _round in range(rounds):
    for label in steady:
      blockStarts.add(index)
      index += len(program.block(label).body)
  roundLength = max(1, index // rounds)
  for position, inst in enumerate(body):
    if position in blockStarts:
      marks.append(clock)
      counters.append((raw, war, dsq, busy, tdm, tdmIssue))
    shift = (position // roundLength) * trip
    if isinstance(inst, Mma):
      srcReady = max([ready.get(_tileKey(ref), 0.0) for ref in inst.srcs
                      if ref.tile.space == "register"] or [0.0])
      start = max(clock, srcReady, pipeEnd)
      if len(pipeEnds) >= wmmaQueue:
        start = max(start, pipeEnds[-wmmaQueue])
      if srcReady > max(clock, pipeEnd):
        raw += srcReady - max(clock, pipeEnd)
      pipeEnd = start + perWmma
      pipeEnds.append(pipeEnd)
      busy += perWmma
      for ref in inst.srcs:
        if ref.tile.space == "register":
          lastRead[_slotKey(ref)] = start
      # An s_wait on the source blocks the in-order stream until the data is back.
      clock = max(clock, srcReady) + 1
    elif isinstance(inst, Move) and inst.op == "ds_load":
      dst = inst.dsts[0]
      buffers = [key for key in (_bufferKey(ref, shift) for ref in inst.srcs) if key]
      gate = max([landed.get(key, 0.0) for key in buffers] or [0.0])
      if gate > clock:
        tdm += gate - clock
        clock = gate
      gate = lastRead.get(_slotKey(dst), 0.0)
      if gate > clock:
        war += gate - clock
        clock = gate
      operand = dst.tile.operand
      if operand.startswith("MXS"):
        count, bytesPerLane = 1, 4
      else:
        side = "A" if operand.endswith("A") else "B"
        dataType = kernel["ProblemType"].get("DataType" + side) or kernel["ProblemType"]["DataType"]
        bytesPerLane = max(4, min(16, int(kernel.get("LocalReadVectorWidth" + side, 4))
                              * dataType.numBytes()))
        count = max(1, ceil(int(dst.size_regs or 1) * 4 / bytesPerLane))
      latency = costs.dsLoadLatency(bytesPerLane)
      done = clock
      for _ in range(count):
        if len(inflight) >= costs.dsQueueDepth:
          oldest = inflight.pop(0)
          if oldest > clock:
            dsq += oldest - clock
            clock = oldest
        start = max(clock, ldsFree)
        ldsFree = start + costs.dsCyclesPerLoad
        done = start + latency + (waves - 1)
        inflight.append(done)
        clock += 1
      ready[_tileKey(dst)] = done
      for key in buffers:
        drained[key] = max(drained.get(key, 0.0), done)
    elif isinstance(inst, Move) and inst.op == "tensor_load":
      buffers = [(key, ref.gen.ring) for key, ref in
                 ((_bufferKey(ref, shift), ref) for ref in inst.dsts) if key]
      gate = max([drained.get((key[0], key[1] - ring), 0.0) for key, ring in buffers] or [0.0])
      if gate > clock:
        tdm += gate - clock
        clock = gate
      for key, _ring in buffers:
        landed[key] = clock + costs.tdmLatency
      clock += costs.tdmIssueCycles
      tdmIssue += costs.tdmIssueCycles
    elif isinstance(inst, Move):
      clock += 1
  marks.append(max(clock, pipeEnd))
  counters.append((raw, war, dsq, busy, tdm, tdmIssue))
  # The LAST round is the steady state: the first ones fill the queues.
  first, last = (rounds - 1) * trip, rounds * trip
  span = (marks[last] - marks[first]) / float(trip)
  delta = [(counters[last][i] - counters[first][i]) / float(trip) for i in range(6)]
  return Timeline(span + costs.tripOverhead, delta[3], delta[0], delta[1], delta[2], trip,
                  delta[4], delta[5])


def lower(theta):
  """Decode θ and lower it to GIR through the production pass pipeline."""
  from ....LoopModel.decoder import decode_theta
  from ....Lowering.loopir_to_gir import lower_to_gir
  from ....Lowering.gir.passes import pipeline as girPipeline
  from ....Lowering.gir.passes import run_pipeline

  decoded = decode_theta(theta, verify=False)
  program = lower_to_gir(decoded.program, decoded.profile)
  run_pipeline(program, passes=girPipeline())
  return program
