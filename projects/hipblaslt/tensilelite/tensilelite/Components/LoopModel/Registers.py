# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Calculate LoopModel register parameters and Valu-ring allocation."""

from typing import NamedTuple

from ...LoopModel import traversal as _geometry

from . import Order


class ValuAlloc(NamedTuple):
  """One operand's Valu allocation and the terms it is made of.

  ``perBuffer * va`` is the size; the rest is the same statement spelled out,
  for the IR dump and for anyone asking why the size is what it is.
  """
  perBuffer: int    #: registers in one buffer -- the prefetch unit after VG
  va: int           #: VA, the buffer count
  puTiles: int      #: PU in instruction tiles, after a TDMSplit that cut it
  puRegs: int       #: PU in registers, before VG divides it
  vg: int           #: VG, the divisor of the unit

  @property
  def total(self) -> int:
    return self.perBuffer * self.va


_READ_AXES = {"A": frozenset(("M", "K")), "B": frozenset(("N", "K"))}


def calculate(kernel):
  """Calculate the concrete per-operand PLR, CLR, VG and VA values."""
  if not kernel.get("UseLoopModel"):
    return

  def value(name, default=-1):
    result = kernel.get(name, default)
    if result is None:
      return int(default)
    try:
      return int(result)
    except (TypeError, ValueError):
      return int(default)

  shape = Order.axes(kernel)
  axes = shape["axes"]
  extents = dict(axes)
  walked = tuple(
      (name, extent)
      for name, extent in axes
      if extent > 1
      or extents.get(name.replace("_inner", "_split"), 1) > 1
  )
  # THE FACTOR AXES, not the storage ones: an extra axis structures the walk whether or not
  # TDMSplit put a region behind it, so the unit, the rotation and the caps follow it.
  factor_axes = shape.get("factor_axes") or shape["regions"]
  all_regions = set(factor_axes["A"]) | set(factor_axes["B"])
  global_outer = next(
      (name.split("_", 1)[0] for name, _extent in walked
       if name not in all_regions),
      walked[0][0].split("_", 1)[0],
  )
  requested_plr = max(0, value("PrefetchLocalRead", 0))
  scalar_clr = max(0, value("ClusterLocalRead", 0))
  values = {}

  for side in ("A", "B"):
    roles = _READ_AXES[side]
    regions = set(factor_axes[side])
    varying = tuple(
        (name, extent)
        for name, extent in walked
        if name.split("_", 1)[0] in roles
    )
    role_modes = tuple(item for item in varying if item[0] not in regions) \
        or varying
    outer_role = role_modes[0][0].split("_", 1)[0]
    inner_role = role_modes[-1][0].split("_", 1)[0]
    reloads = outer_role != global_outer
    original_unit = tuple(
        item for item in varying
        if item[0].split("_", 1)[0] == inner_role
    )
    unit_modes = tuple(item for item in original_unit if item[0] not in regions)
    unit = _geometry.product(extent for _name, extent in unit_modes)
    original_tiles = _geometry.product(
        extent for _name, extent in original_unit
    )
    split_unit = max(1, original_tiles // max(1, unit))

    vg_name, va_name = "VgprGroup" + side, "VgprAlloc" + side
    explicit_vg = value(vg_name) != -1
    explicit_va = value(va_name) != -1
    if explicit_vg or explicit_va:
      kernel["SuppressNoLoadLoop"] = True
    vg = max(1, value(vg_name, 1))
    if unit % vg:
      raise RuntimeError(
          "VgprGroup%s=%d does not divide %s's %d-tile prefetch unit"
          % (side, vg, side, unit))

    role_extents = {
        role: _geometry.product(
            extent for name, extent in role_modes
            if name.split("_", 1)[0] == role
        )
        for role in roles
    }
    cap_role = next(
        (name.split("_", 1)[0] for name, _extent in role_modes
         if role_extents[name.split("_", 1)[0]] > 1),
        outer_role,
    )
    cap = max(1, role_extents[cap_role] * vg)
    plr_name = "PrefetchLocalRead" + side
    plr = value(plr_name)
    if plr == -1:
      plr = min(requested_plr, cap)
    elif plr > cap:
      raise RuntimeError(
          "%s=%d exceeds %d, the most %s can run ahead"
          % (plr_name, plr, cap, side))

    clr_name = "ClusterLocalRead" + side
    clr = value(clr_name)
    auto_clr = clr == -1
    if auto_clr:
      clr = 0 if explicit_vg or explicit_va else scalar_clr
    if reloads and auto_clr and not (explicit_vg or explicit_va):
      clr = 1

    va = value(va_name)
    if va == -1:
      rotation = varying if reloads else tuple(
          item for item in varying
          if item[0].split("_", 1)[0] != outer_role
      )
      summation = tuple(
          item for item in varying
          if item[0].split("_", 1)[0] == "K"
      )
      enumerator = tuple(
          item for item in varying
          if item[0].split("_", 1)[0] == outer_role
      )
      unpacked = _geometry.product(
          extent for name, extent in unit_modes
          if name not in {mode for mode, _extent in rotation}
      )
      level = next(
          (name for name, _extent in walked if name not in all_regions),
          None,
      )
      reduction_ring = tuple(
          item for item in rotation
          if item[0].split("_", 1)[0] == "K"
      ) if level and level.startswith("K_") and unpacked > 1 else ()
      ring = reduction_ring or (summation if reloads else enumerator)
      rotation_regions = regions & {name for name, _extent in rotation}
      positions = max(
          1,
          _geometry.product(
              extent for name, extent in ring
              if name not in rotation_regions
          ),
      )
      if clr == 0:
        if split_unit > 1 or vg > 1:
          va = plr + 1
        else:
          read_phi = kernel.get("ReadPhi") or {}
          folded = int(read_phi.get(side, 1) or 1) > 1
          broadcast_role = "N" if side == "A" else "M"
          positions_by_name = {
              name: index for index, (name, _extent) in enumerate(walked)
          }
          ring_at = [positions_by_name[name] for name, _extent in ring]
          unit_at = [positions_by_name[name] for name, _extent in unit_modes]
          folded_across_broadcast = folded and ring_at and unit_at and any(
              min(ring_at) < index < max(unit_at)
              for index, (name, _extent) in enumerate(walked)
              if name.split("_", 1)[0] == broadcast_role
          )
          floor = positions if folded_across_broadcast else 1
          want = max(plr + 1, floor)
          va = next(
              (width for width in range(want, positions + 1)
               if positions % width == 0),
              positions,
          )
      else:
        resident_tiles = _geometry.product(
            extent for _name, extent in varying
        )
        if resident_tiles % unit:
          raise RuntimeError(
              "%s's %d resident tiles do not divide into %d-tile "
              "prefetch units" % (side, resident_tiles, unit))
        va = max(1, resident_tiles // unit * vg)
    if va < 1:
      raise RuntimeError("%s=%d must be positive" % (va_name, va))

    values[side] = (max(0, plr), max(0, clr), vg, int(va), explicit_va)

  for side, (plr, clr, vg, va, explicit_va) in values.items():
    kernel["PrefetchLocalRead" + side] = plr
    kernel["ClusterLocalRead" + side] = clr
    kernel["VgprGroup" + side] = vg
    kernel["VgprAlloc" + side] = va
    kernel["VgprAllocExplicit" + side] = explicit_va
  kernel["PrefetchLocalRead"] = max(value[0] for value in values.values())
  kernel["ClusterLocalRead"] = min(value[1] for value in values.values())


def operandValuPuVaVg(writer, kernel, tc, bpe) -> ValuAlloc:
  """Registers in one buffer, and the buffer count, for one UseLoopModel operand.

  The operand's final Valu size is ``perBuffer * va``, which is PU * VA / VG.
  Nothing else scales it.

  PU is the innermost of the operand's loop axes: KMN gives A=M and B=N,
  MKN gives A=K and B=N.  Its register count is that many instruction tiles.
  TDMSplit changes that PU first; VG then divides the resulting per-region PU
  exactly as stated by VgprGroupA/B. ``calculate`` stores PU, VG and VA before
  allocation.
  """
  side = "A" if str(tc).endswith("A") else "B"
  from .Theta import loopModelTheta
  theta, _depths = loopModelTheta(writer, kernel)
  pu = _geometry.prefetch_unit_tiles(theta, theta.op(str(tc)))
  vg = max(1, int(kernel["VgprGroup" + side]))
  iptName = "MIInputPerThread" + (tc if bpe is None else side)
  ipt = kernel[iptName]
  puRegs = (int(pu * ipt // writer.states.bpr) if bpe is None
            else int(pu * ipt * bpe // writer.states.bpr))
  if puRegs % vg:
    raise RuntimeError(
        "VgprGroup%s=%d does not divide %s's %d-register prefetch unit" % (side, vg, tc, puRegs))
  perBuffer = puRegs // vg

  operand = theta.op(str(tc))
  if bpe is None:
    required = _geometry.register_layout(theta, _depths)[str(tc)]["total"]
    va = max(1, -(-required // max(1, perBuffer)))
  else:
    va = int(kernel["VgprAlloc" + side])
  if va < 1:
    raise RuntimeError(
        "VgprAlloc%s must be calculated before register allocation" % side)

  direct = kernel.get("DirectToVgpr" + tc, kernel.get("DirectToVgpr" + side, False))
  packing = ((writer.states.packDTVA or writer.states.convDTVA) if side == "A"
             else (writer.states.packDTVB or writer.states.convDTVB))
  if direct and not packing:
    return ValuAlloc(0, va, pu, puRegs, vg)
  return ValuAlloc(perBuffer, va, pu, puRegs, vg)


def operandNumVgprBuffer(writer, tc):
  """Valu ring depth for one operand.

  vgprAllocationImplClassic stores the per-operand count.  A zero field means
  allocation has not run, so the shared scaffold count still applies.
  """
  stored = {"A": writer.states.numVgprBufferA,
            "B": writer.states.numVgprBufferB,
            "MXSA": writer.states.numVgprBufferMXSA,
            "MXSB": writer.states.numVgprBufferMXSB}[tc]
  return int(stored or writer.states.numVgprBuffer)


def loopModelRegBuffers(writer, kernel):
  """Per-operand ring depth -- that operand's stated VGPR buffer count.

  GIR and the `Valu*_X*` table read the same cached θ/S, so every slot GIR names has a matching
  symbol.  Register groups of one operand share the X axis, so that operand's depth is the max
  over its groups.  A, B, and the MX scales do not share a depth; the caller keeps one count
  per operand.
  """
  if not kernel["UseLoopModel"]:
    return {}
  from .Theta import loopModelTheta
  theta, S = loopModelTheta(writer, kernel)
  widths = {}
  for op in theta.operands:
    if op.trajectory.fragment_fill is None:
      continue
    w = max((_geometry.group_ring_depth(theta, op, group, S)
             for group in op.fragment.groups()), default=1)
    widths[op.name] = max(1, int(w))
  return widths


def loopModelValuRegs(writer, kernel):
  """Physical VGPRs per tensorChar for the Valu rings GIR names.

  Each register group owns a compact block for every one of its W rotation slots.
  """
  if not kernel["UseLoopModel"]:
    return {}
  from .Theta import loopModelTheta
  theta, S = loopModelTheta(writer, kernel)
  out = {}
  for op in theta.operands:
    if op.trajectory.fragment_fill is None:
      continue
    out[op.name] = int(_geometry.operand_emitted_regs(theta, op, S))
  return out


def loopModelRegisterLayout(writer, kernel):
  """Compact per-group physical register offsets from the cached θ/S."""
  if not kernel["UseLoopModel"]:
    return {}
  from .Theta import loopModelTheta
  theta, S = loopModelTheta(writer, kernel)
  return _geometry.register_layout(theta, S)


def applyLoopModelValuRegs(writer, kernel):
  """Overwrite the scaffold's Valu-ring sizing with θ/S's, after it has run."""
  if not kernel["UseLoopModel"]:
    return
  required = writer.states.loopModelValuVgprs
  byOperand = {"A": writer.states.a, "B": writer.states.b}
  if kernel["ProblemType"]["MXBlockA"]:
    byOperand["MXSA"] = writer.states.mxsa
  if kernel["ProblemType"]["MXBlockB"]:
    byOperand["MXSB"] = writer.states.mxsb
  missing = [name for name in byOperand if name not in required]
  if missing:
    raise RuntimeError("UseLoopModel has no Valu VGPR allocation for %s" % ", ".join(missing))
  for name, state in byOperand.items():
    count = int(required[name])
    if count < 1:
      raise RuntimeError("UseLoopModel derived invalid Valu VGPR count %d for %s" % (count, name))
    state.numVgprValu = count
