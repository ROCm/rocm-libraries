# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import List, Tuple

from .SubtileGeometry import ABLRGeometry, LRTag_TLU1
from .SubtileLREmit import (
    _allocLROffsetRegisters,
    _deallocLROffsetRegisters,
    _emitLocalReadOffset,
    _emitLocalRead,
    _emitLRDTLInit,
    _emitLRLDSBufferSwap,
)


class ABLRTile:
  """Mutable LR tile for A/B local reads.

  Holds any frozen ABLRGeometry config. Shape-dependent parameters are
  computed once in __init__ from the config; emit methods read those
  parameters directly with no isinstance branching.
  """

  def __init__(self, config: ABLRGeometry):
    self.config = config
    self.sharedVgprLROffset: List[int] = []
    self.sharedVgprLROffsetSwap: List[int] = []
    self.localSubtiles: List = []

    # Shape descriptor — same convention as ABGRTile.
    if isinstance(config.tag, LRTag_TLU1):
      self.contiguousDim      = 'M'
      self.contiguousElements = config.loadShape.m
    else:  # row-major
      self.contiguousDim      = 'K'
      self.contiguousElements = config.loadShape.k

  @property
  def subtileShape(self) -> Tuple[int, int]:
    return self.config.subtileShape

  @property
  def loadShape(self):
    return self.config.loadShape

  # --- Register allocation ---

  def allocOffsetRegisters(self, ti, writer, kernel):
    return _allocLROffsetRegisters(self.config.tag, self, ti, writer, kernel)

  def deallocOffsetRegisters(self, ti, writer, kernel):
    return _deallocLROffsetRegisters(self.config.tag, self, ti, writer, kernel)

  # --- Emit ---

  def emitLocalReadOffset(self, ti, writer, kernel):
    return _emitLocalReadOffset(self.config.tag, self, ti, writer, kernel)

  def emitLocalRead(self, ti, writer, kernel):
    return _emitLocalRead(self.config.tag, self, ti, writer, kernel)

  def emitDTLInit(self, ti, writer, kernel):
    return _emitLRDTLInit(self.config.tag, self, ti, writer, kernel)

  def emitLDSBufferSwap(self, ti, writer, kernel):
    return _emitLRLDSBufferSwap(self.config.tag, self, ti, writer, kernel)
