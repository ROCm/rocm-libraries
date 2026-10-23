// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The kernel body before the K loop: arguments, buffer resources, the skinny
// dimension dispatch, the tile and K rotation (and, in the 3-D launch, the
// XCD regrouping of the tiles), and the per-lane bases.

#include "SkinnyGEMMEmitter.hpp"

#include "container.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace rocm::tensilelite::codegen::detail {

namespace {

constexpr int selectorMaskStep = 0x01010101;

bool isScalar(RegisterRef ref) { return ref.kind == RegisterKind::sgpr; }

// Bytes of an operand of cols columns: (cols - 1) * ld + alignUp(k, 4), as
// the last column is padded to 4 bytes, not to ld.
void operandBytes(IRBuilder &builder, RegisterRef dst, RegisterRef cols,
                  RegisterRef ld, RegisterRef alignedK) {
  builder.addi32(dst, cols, -1);
  builder.muli32(dst, dst, ld);
  builder.addi32(dst, dst, alignedK);
}

// Bytes of the scale matrix of cols rows: ceilDiv(cols, 32) row groups.
void scaleBytes(IRBuilder &builder, RegisterRef dst, RegisterRef cols,
                RegisterRef rowGroup) {
  builder.addi32(dst, cols, 31);
  shiftRight(builder, dst, dst, 5);
  builder.muli32(dst, dst, rowGroup);
}

// The scale base of a column of a row group matrix: (col / 32) * rowGroup +
// (col mod 16) * 4, before the lane's role offset. Overwrites col.
void scaleRowOffset(IRBuilder &builder, RegisterRef dst, RegisterRef col,
                    RegisterRef rowGroup) {
  shiftRight(builder, dst, col, 5);
  builder.muli32(dst, dst, rowGroup);
  bitAnd(builder, col, col, 15);
  shiftLeft(builder, col, col, 2);
  builder.addi32(dst, dst, col);
}

// dst = SCC ? ifSet : otherwise for one register or an aligned pair.
void selectOnScc(IRBuilder &builder, RegisterRef dst, RegisterRef ifSet,
                 RegisterRef otherwise) {
  if (dst.count == 2)
    builder.create<rocisa::SCSelectB64>(dst.container(), ifSet.container(),
                                        otherwise.container());
  else
    builder.create<rocisa::SCSelectB32>(dst.container(), ifSet.container(),
                                        otherwise.container());
}

} // namespace

bool isPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

int log2Exact(int value) {
  if (!isPowerOfTwo(value))
    throw std::logic_error("value is not a power of two");
  int log = 0;
  while ((1 << log) < value)
    ++log;
  return log;
}

// Hacker's Delight, signed magic numbers for a divisor d >= 2.
SignedMagic signedMagic(int32_t d) {
  constexpr uint32_t two31 = 0x80000000u;
  uint32_t ad = static_cast<uint32_t>(d);
  uint32_t t = two31;
  uint32_t anc = t - 1 - t % ad;
  int p = 31;
  uint32_t q1 = two31 / anc;
  uint32_t r1 = two31 - q1 * anc;
  uint32_t q2 = two31 / ad;
  uint32_t r2 = two31 - q2 * ad;
  uint32_t delta = 0;
  do {
    ++p;
    q1 *= 2;
    r1 *= 2;
    if (r1 >= anc) {
      ++q1;
      r1 -= anc;
    }
    q2 *= 2;
    r2 *= 2;
    if (r2 >= ad) {
      ++q2;
      r2 -= ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  return {static_cast<int32_t>(q2 + 1), p - 32};
}

void divideByConstant(IRBuilder &builder, RegisterRef quotient,
                      RegisterRef dividend, int divisor) {
  SignedMagic magic = signedMagic(divisor);
  RegisterRange sign = builder.allocateSgpr(1);
  builder.create<rocisa::SMulHII32>(quotient.container(), dividend.container(),
                                    magic.multiplier);
  if (magic.multiplier < 0)
    builder.addi32(quotient, quotient, dividend);
  if (magic.shift > 0)
    builder.create<rocisa::SAShiftRightI32>(quotient.container(), magic.shift,
                                            quotient.container());
  shiftRight(builder, sign.ref(), quotient, 31);
  builder.addi32(quotient, quotient, sign.ref());
}

SkinnyDims::SkinnyDims(const SkinnyGEMMConfig &config)
    : bm(config.bm), tn(config.tn), lanesK(config.lanesK), waves(config.waves),
      prefetch(config.prefetch), grid3d(config.grid3d), group(config.group),
      numXcds(config.numXcds),
      reordered(config.group != 1 || config.numXcds != 1),
      upIters(config.upIters), partialSteps(config.partialSteps) {
  columns = bm + tn;
  accumulators = bm * tn;
  columnGroups = 64 / lanesK;
  tileWide = waves * columnGroups * tn;
  xcdRun = isPowerOfTwo(tileWide) && tileWide < scaleRowColumns
               ? scaleRowColumns / tileWide
               : 1;
  ki = 64 * lanesK;
  subgroupLanes = std::min(lanesK, 8);
  halfSubgroup = subgroupLanes / 2;
  log2LanesK = log2Exact(lanesK);
  log2ColumnGroups = log2Exact(columnGroups);
  log2Subgroup = log2Exact(subgroupLanes);
  log2Half = log2Exact(halfSubgroup);
  log2Ki = log2Exact(ki);
  slots = std::max(prefetch, 1);
  scalesShared = upIters != 0 && waves > 1 && columnGroups < 4;
  roleSlots = scalesShared ? waves * columnGroups : columnGroups;
  scaleLoads = upIters != 0 ? 0 : columns;
  loadsPerSlot = scaleLoads + chunks * columns;
  ldsBytes = scalesShared ? upIters * columns * waves * 256 : 0;
  chunkStride = 16 * subgroupLanes;
}

void shiftLeft(IRBuilder &builder, RegisterRef dst, RegisterRef src,
               int amount) {
  if (amount == 0) {
    if (dst != src)
      builder.movb32(dst, src);
    return;
  }
  if (isScalar(dst))
    builder.create<rocisa::SLShiftLeftB32>(dst.container(), amount,
                                           src.container());
  else
    builder.create<rocisa::VLShiftLeftB32>(dst.container(), amount,
                                           src.container());
}

void shiftRight(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                int amount) {
  if (amount == 0) {
    if (dst != src)
      builder.movb32(dst, src);
    return;
  }
  if (isScalar(dst))
    builder.create<rocisa::SLShiftRightB32>(dst.container(), amount,
                                            src.container());
  else
    builder.create<rocisa::VLShiftRightB32>(dst.container(), amount,
                                            src.container());
}

void bitAnd(IRBuilder &builder, RegisterRef dst, RegisterRef src, int mask) {
  if (isScalar(dst))
    builder.create<rocisa::SAndB32>(dst.container(), mask, src.container());
  else
    builder.create<rocisa::VAndB32>(dst.container(), mask, src.container());
}

void multiplyBy(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                int factor) {
  if (isPowerOfTwo(factor)) {
    shiftLeft(builder, dst, src, log2Exact(factor));
    return;
  }
  builder.muli32(dst, src, factor);
}

int loadsIssuedAfter(const SkinnyDims &dims, SlotLoadPosition where) {
  int sameSlot = dims.loadsPerSlot - 1 - where.position;
  return sameSlot + where.groupsAfter * dims.loadsPerSlot + where.ownRefill;
}

SkinnyGEMMEmitter::SetupScalars::SetupScalars(IRBuilder &builder, bool grid3d)
    : blocksX(builder.allocateSgpr(1)),
      blocksY(grid3d ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                     : std::nullopt),
      rowGroup(builder.allocateSgpr(1)), ldS(builder.allocateSgpr(1)),
      ldW(builder.allocateSgpr(1)),
      splits(grid3d ? std::nullopt
                    : std::optional<RegisterRange>(builder.allocateSgpr(1))) {}

SkinnyGEMMEmitter::ResourceScalars::ResourceScalars(IRBuilder &builder,
                                                    bool grid3d)
    : ldD(builder.allocateSgpr(1)), ldA(builder.allocateSgpr(1)),
      ldB(builder.allocateSgpr(1)), m(builder.allocateSgpr(1)),
      n(builder.allocateSgpr(1)),
      blocksZ(grid3d ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                     : std::nullopt),
      aPtr(builder.allocateSgpr(2, 2)), bPtr(builder.allocateSgpr(2, 2)),
      sclAPtr(builder.allocateSgpr(2, 2)), sclBPtr(builder.allocateSgpr(2, 2)),
      aBytes(builder.allocateSgpr(1)), bBytes(builder.allocateSgpr(1)),
      sclABytes(builder.allocateSgpr(1)), sclBBytes(builder.allocateSgpr(1)) {}

SkinnyGEMMEmitter::SkinnyGEMMEmitter(Kernel &kernel,
                                     const SkinnyGEMMConfig &config,
                                     const SkinnyArguments &arguments)
    : kernel(kernel), builder(kernel.builder()), dims(config),
      arguments(arguments),
      ringData(vgpr(dims.slots * SkinnyDims::chunks * dims.columns * 4, 2)),
      ringRaw(dims.upIters == 0 ? std::optional<RegisterRange>(
                                      vgpr(dims.slots * dims.columns))
                                : std::nullopt),
      w0(uniformWide() ? sgpr() : vgpr()), acc(vgpr(dims.accumulators)),
      dBase(vgpr(dims.columns)), sBase(vgpr(dims.columns)),
      selectors(vgpr((16 % dims.bm == 0 ? 1 : dims.bm) +
                     (16 % dims.tn == 0 ? 1 : dims.tn))),
      addresses(vgpr(dims.scalesShared ? 1 : 2)),
      ldsStage(dims.scalesShared ? std::optional<RegisterRange>(vgpr())
                                 : std::nullopt),
      scaleGroup(dims.partialSteps != 0 ? std::optional<RegisterRange>(vgpr())
                                        : std::nullopt),
      sRsrc(sgpr(4, 4)), wRsrc(sgpr(4, 4)), sSclRsrc(sgpr(4, 4)),
      wSclRsrc(sgpr(4, 4)), dRsrc(sgpr(4, 4)), wsRsrc(sgpr(4, 4)), k(sgpr()),
      spk(sgpr()), mn(sgpr()), dStrideS(sgpr()), dStrideW(sgpr()),
      wsStrideS(sgpr()), wsStrideW(sgpr()), skinnyCols(sgpr()),
      wideCols(sgpr()), contigS(sgpr()), s0(sgpr()), nIterFull(sgpr()),
      fitRot(dims.upIters != 1 ? std::optional<RegisterRange>(sgpr())
                               : std::nullopt),
      kBaseIter(sgpr()),
      splitStart(dims.partialSteps == 2 ? std::optional<RegisterRange>(sgpr())
                                        : std::nullopt),
      splitEnd(dims.partialSteps != 0 ? std::optional<RegisterRange>(sgpr())
                                      : std::nullopt),
      startIter(dims.partialSteps == 2 ? std::optional<RegisterRange>(sgpr())
                                       : std::nullopt),
      vecOff(dims.partialSteps == 2 && dims.upIters >= 2
                 ? std::optional<RegisterRange>(sgpr())
                 : std::nullopt),
      chunkBase(sgpr(4)), chunkBaseReady(4, false),
      sValid(sgpr(2 * dims.bm, 2)), wValid(sgpr(2 * dims.tn, 2)) {}

RegisterRange SkinnyGEMMEmitter::sgpr(int count, int alignment) {
  return builder.allocateSgpr(count, alignment);
}

RegisterRange SkinnyGEMMEmitter::vgpr(int count, int alignment) {
  return builder.allocateVgpr(count, alignment);
}

// With one wave of 64 lanes in one cluster the wide column of every lane is
// the same, so it stays in an SGPR.
bool SkinnyGEMMEmitter::uniformWide() const {
  return dims.waves == 1 && dims.lanesK == 64;
}

void SkinnyGEMMEmitter::emit() {
  if (!dims.grid3d) {
    emitPersistent();
    return;
  }
  tile = TileIds{kernel.workgroupIdX().ref(), kernel.workgroupIdY().ref(),
                 kernel.workgroupIdZ().ref()};
  {
    SetupScalars scalars(builder, /*grid3d=*/true);
    {
      ResourceScalars resources(builder, /*grid3d=*/true);
      loadArguments(scalars, resources);
      makeResources(scalars, resources);
    }
    computeTile(scalars);
    LaneValues lane = computeLaneValues();
    fillDataBases(scalars, lane);
    fillScaleBases(scalars, lane, lane.w0Load->ref());
    fillSelectors();
    zeroAccumulators();
  }
  runTile();
  emitEpilogue();
  builder.createEndProgram();
}

// The pointers load straight into the words 0 and 1 of the resources of D and
// the workspace; the others wait in pairs for the path dispatch.
void SkinnyGEMMEmitter::loadArguments(SetupScalars &scalars,
                                      ResourceScalars &resources) {
  if (resources.blocksZ)
    kernel.loadArgument(resources.blocksZ->ref(), arguments.blocksZ);
  kernel.loadArgument(scalars.blocksX.ref(), arguments.blocksX);
  if (scalars.blocksY)
    kernel.loadArgument(scalars.blocksY->ref(), arguments.blocksY);
  kernel.loadArgument(spk.ref(), arguments.spk);
  kernel.loadArgument(k.ref(), arguments.k);
  kernel.loadArgument(resources.n.ref(), arguments.n);
  kernel.loadArgument(resources.m.ref(), arguments.m);
  kernel.loadArgument(resources.ldB.ref(), arguments.ldB);
  kernel.loadArgument(resources.ldA.ref(), arguments.ldA);
  kernel.loadArgument(resources.ldD.ref(), arguments.ldD);
  kernel.loadArgument(wsRsrc.ref(0, 2), arguments.ws);
  kernel.loadArgument(resources.sclBPtr.ref(), arguments.scalesB);
  kernel.loadArgument(resources.sclAPtr.ref(), arguments.scalesA);
  kernel.loadArgument(resources.bPtr.ref(), arguments.b);
  kernel.loadArgument(resources.aPtr.ref(), arguments.a);
  kernel.loadArgument(dRsrc.ref(0, 2), arguments.d);
  // Scalar loads return out of order, so only lgkmcnt(0) guarantees any of
  // them.
  builder.createWaitCnt(WaitCounts{.scalarMemory = 0});
}

void SkinnyGEMMEmitter::makeResources(const SetupScalars &scalars,
                                      const ResourceScalars &resources) {
  makeBufferSizes(scalars, resources);
  dispatchPaths(scalars, resources);
  builder.makeRsrc(dRsrc.ref(), dRsrc.ref(0, 2), dRsrc.ref(2));
  builder.makeRsrc(wsRsrc.ref(), wsRsrc.ref(0, 2), wsRsrc.ref(2));
}

void SkinnyGEMMEmitter::makeBufferSizes(const SetupScalars &scalars,
                                        const ResourceScalars &resources) {
  RegisterRange alignedK = sgpr();
  builder.addi32(alignedK.ref(), k, 3);
  shiftRight(builder, alignedK.ref(), alignedK.ref(), 2);
  shiftLeft(builder, alignedK.ref(), alignedK.ref(), 2);
  operandBytes(builder, resources.aBytes.ref(), resources.m.ref(),
               resources.ldA.ref(), alignedK.ref());
  operandBytes(builder, resources.bBytes.ref(), resources.n.ref(),
               resources.ldB.ref(), alignedK.ref());
  // A row group of the scales holds k rounded up to 256 bytes.
  builder.addi32(scalars.rowGroup.ref(), k, 255);
  shiftRight(builder, scalars.rowGroup.ref(), scalars.rowGroup.ref(), 8);
  shiftLeft(builder, scalars.rowGroup.ref(), scalars.rowGroup.ref(), 8);
  scaleBytes(builder, resources.sclABytes.ref(), resources.m.ref(),
             scalars.rowGroup.ref());
  scaleBytes(builder, resources.sclBBytes.ref(), resources.n.ref(),
             scalars.rowGroup.ref());
  // D holds ((n - 1) * ldD + m) bf16 elements, as (2n - 2) * ldD + 2m bytes.
  RegisterRef dBytes = dRsrc.ref(2);
  shiftLeft(builder, dBytes, resources.n.ref(), 1);
  builder.addi32(dBytes, dBytes, -2);
  builder.muli32(dBytes, dBytes, resources.ldD.ref());
  shiftLeft(builder, alignedK.ref(), resources.m.ref(), 1);
  builder.addi32(dBytes, dBytes, alignedK.ref());
  // The workspace holds splits * m * n f32 elements; the splits are the
  // blocks of the grid's z dimension, or ceilDiv(k, spk) in the 1-D launch.
  RegisterRef splits =
      resources.blocksZ ? resources.blocksZ->ref() : scalars.splits->ref();
  if (!resources.blocksZ)
    computeSplits(splits);
  RegisterRef wsBytes = wsRsrc.ref(2);
  builder.muli32(wsBytes, resources.m.ref(), splits);
  builder.muli32(wsBytes, wsBytes, resources.n.ref());
  shiftLeft(builder, wsBytes, wsBytes, 2);
  builder.muli32(mn.ref(), resources.m.ref(), resources.n.ref());
}

// splits = ceilDiv(k, spk) with the exact signed division, as the reference
// divides at run time.
void SkinnyGEMMEmitter::computeSplits(RegisterRef splits) {
  builder.addi32(splits, k.ref(), spk.ref());
  builder.addi32(splits, splits, -1);
  builder.divsi32(splits, splits, spk.ref(), /*useIEEE=*/true);
}

// Selects the operands, strides and extents of the skinny and the wide
// dimension on m <= n. The s_cselects follow one s_cmp and no SALU between
// them writes SCC.
void SkinnyGEMMEmitter::dispatchPaths(const SetupScalars &scalars,
                                      const ResourceScalars &resources) {
  builder.create<rocisa::SCmpLeU32>(resources.m.container(),
                                    resources.n.container());
  selectOnScc(builder, sRsrc.ref(0, 2), resources.aPtr.ref(),
              resources.bPtr.ref());
  selectOnScc(builder, sRsrc.ref(2), resources.aBytes.ref(),
              resources.bBytes.ref());
  selectOnScc(builder, wRsrc.ref(0, 2), resources.bPtr.ref(),
              resources.aPtr.ref());
  selectOnScc(builder, wRsrc.ref(2), resources.bBytes.ref(),
              resources.aBytes.ref());
  selectOnScc(builder, sSclRsrc.ref(0, 2), resources.sclAPtr.ref(),
              resources.sclBPtr.ref());
  selectOnScc(builder, sSclRsrc.ref(2), resources.sclABytes.ref(),
              resources.sclBBytes.ref());
  selectOnScc(builder, wSclRsrc.ref(0, 2), resources.sclBPtr.ref(),
              resources.sclAPtr.ref());
  selectOnScc(builder, wSclRsrc.ref(2), resources.sclBBytes.ref(),
              resources.sclABytes.ref());
  selectOnScc(builder, scalars.ldS.ref(), resources.ldA.ref(),
              resources.ldB.ref());
  selectOnScc(builder, scalars.ldW.ref(), resources.ldB.ref(),
              resources.ldA.ref());
  builder.create<rocisa::SCSelectB32>(dStrideS.container(), 1,
                                      resources.ldD.container());
  builder.create<rocisa::SCSelectB32>(dStrideW.container(),
                                      resources.ldD.container(), 1);
  builder.create<rocisa::SCSelectB32>(wsStrideS.container(), 1,
                                      resources.m.container());
  builder.create<rocisa::SCSelectB32>(wsStrideW.container(),
                                      resources.m.container(), 1);
  builder.create<rocisa::SCSelectB32>(contigS.container(), 1, 0);
  builder.create<rocisa::SMinU32>(
      skinnyCols.container(), resources.m.container(), resources.n.container());
  builder.create<rocisa::SMaxU32>(wideCols.container(), resources.m.container(),
                                  resources.n.container());
  for (RegisterRange *rsrc : {&sRsrc, &wRsrc, &sSclRsrc, &wSclRsrc})
    builder.makeRsrc(rsrc->ref(), rsrc->ref(0, 2), rsrc->ref(2));
}

// The persistent launch passes the rotation of the tile, the 3-D launch
// computes it from the workgroup ids and then regroups tx per XCD.
void SkinnyGEMMEmitter::computeTile(const SetupScalars &scalars,
                                    std::optional<RegisterRef> rotation) {
  RegisterRef ty = tile.ty;
  RegisterRef kz = tile.kz;
  // With partial steps the bounds outlive the setup, in members.
  std::optional<RegisterRange> startLocal;
  std::optional<RegisterRange> endLocal;
  if (!splitStart)
    startLocal = sgpr();
  if (!splitEnd)
    endLocal = sgpr();
  RegisterRef ks = splitStart ? splitStart->ref() : startLocal->ref();
  RegisterRef ke = splitEnd ? splitEnd->ref() : endLocal->ref();
  // Only the last split can be shorter than spk.
  builder.muli32(ks, kz, spk.ref());
  builder.addi32(ke, ks, spk.ref());
  builder.create<rocisa::SMinI32>(ke.container(), ke.container(),
                                  k.container());
  // The steps need ke itself, so its iteration index goes to a temporary.
  std::optional<RegisterRange> endIterLocal;
  RegisterRef keIter = ke;
  if (splitEnd) {
    endIterLocal = sgpr();
    keIter = endIterLocal->ref();
  }
  builder.create<rocisa::SAShiftRightI32>(keIter.container(), dims.log2Ki,
                                          ke.container());
  builder.addi32(kBaseIter.ref(), ks, dims.ki - 1);
  builder.create<rocisa::SAShiftRightI32>(kBaseIter.container(), dims.log2Ki,
                                          kBaseIter.container());
  builder.subi32(nIterFull.ref(), keIter, kBaseIter.ref());
  builder.create<rocisa::SMaxI32>(nIterFull.container(), nIterFull.container(),
                                  0);
  if (startIter)
    shiftRight(builder, startIter->ref(), ks, dims.log2Ki);
  if (vecOff) {
    // The head step owns entry 0 of the up-front vector when the split starts
    // inside an iteration.
    RegisterRange misaligned = sgpr();
    bitAnd(builder, misaligned.ref(), ks, dims.ki - 1);
    builder.create<rocisa::SCSelectB32>(vecOff->container(), 1, 0);
  }
  if (fitRot) {
    std::optional<RegisterRange> computed;
    if (!rotation) {
      computed = sgpr();
      computeRotation(scalars, computed->ref());
      rotation = computed->ref();
    }
    // The rotation falls back to 0 when it is not below the iteration count,
    // so the physical iteration wraps once without a division.
    builder.create<rocisa::SCmpLtU32>(rotation->container(),
                                      nIterFull.container());
    builder.create<rocisa::SCSelectB32>(fitRot->container(),
                                        rotation->container(), 0);
  }
  multiplyBy(builder, s0.ref(), ty, dims.bm);
  if (dims.grid3d && dims.xcdRun > 1)
    groupTilesPerXcd(scalars);
}

// The xcdRun tiles that share a scale line run on one XCD, so its L2 fetches
// the line once instead of every XCD fetching it. The dispatcher sends
// workgroup x to XCD x mod X, so within each full block of X * xcdRun ids the
// tile is (x mod X) * xcdRun + (x / X) mod xcdRun; the tail of the grid keeps
// the identity so that every tile still runs exactly once.
void SkinnyGEMMEmitter::groupTilesPerXcd(const SetupScalars &scalars) {
  int xcds = kernel.target().xcdCount();
  int span = xcds * dims.xcdRun;
  RegisterRange logical = sgpr();
  RegisterRange term = sgpr();
  shiftRight(builder, logical.ref(), tile.tx, log2Exact(xcds));
  bitAnd(builder, logical.ref(), logical.ref(), dims.xcdRun - 1);
  bitAnd(builder, term.ref(), tile.tx, xcds - 1);
  shiftLeft(builder, term.ref(), term.ref(), log2Exact(dims.xcdRun));
  builder.addi32(logical.ref(), logical.ref(), term.ref());
  bitAnd(builder, term.ref(), tile.tx, -span);
  builder.addi32(logical.ref(), logical.ref(), term.ref());
  bitAnd(builder, term.ref(), scalars.blocksX.ref(), -span);
  // SCC must survive from the compare to the select.
  builder.create<rocisa::SCmpLtU32>(tile.tx.container(), term.container());
  builder.create<rocisa::SCSelectB32>(tile.tx.container(), logical.container(),
                                      tile.tx.container());
}

// rot = (key + key / 16) mod NI with key the linear block id and NI =
// ceilDiv(spk, KI), or the literal UP_ITERS with up-front scales, which bounds
// the iterations of every split. The ring-scale remainder uses the exact
// signed division instead of LLVM's srem expansion.
void SkinnyGEMMEmitter::computeRotation(const SetupScalars &scalars,
                                        RegisterRef rot) {
  RegisterRef tx = tile.tx;
  RegisterRef ty = tile.ty;
  RegisterRef kz = tile.kz;
  RegisterRange key = sgpr();
  builder.muli32(key.ref(), scalars.blocksY->ref(), kz);
  builder.addi32(key.ref(), key.ref(), ty);
  builder.muli32(key.ref(), key.ref(), scalars.blocksX.ref());
  builder.addi32(key.ref(), key.ref(), tx);
  std::optional<RegisterRange> ni;
  if (dims.upIters == 0) {
    ni = sgpr();
    builder.addi32(ni->ref(), spk.ref(), dims.ki - 1);
    shiftRight(builder, ni->ref(), ni->ref(), dims.log2Ki);
  }
  blockRotation(rot, key.ref(),
                ni ? std::optional<RegisterRef>(ni->ref()) : std::nullopt);
}

// rot = (key + key / 16) mod NI, or mod the literal UP_ITERS without ni.
void SkinnyGEMMEmitter::blockRotation(RegisterRef rot, RegisterRef key,
                                      std::optional<RegisterRef> ni) {
  RegisterRange sum = sgpr();
  shiftRight(builder, sum.ref(), key, 4);
  builder.addi32(sum.ref(), sum.ref(), key);
  if (ni) {
    RegisterRange quotient = sgpr();
    builder.divsi32(quotient.ref(), sum.ref(), *ni, /*useIEEE=*/true);
    builder.muli32(quotient.ref(), quotient.ref(), *ni);
    builder.subi32(rot, sum.ref(), quotient.ref());
    return;
  }
  if (isPowerOfTwo(dims.upIters)) {
    builder.create<rocisa::SAndB32>(rot.container(), sum.container(),
                                    dims.upIters - 1);
    return;
  }
  RegisterRange quotient = sgpr();
  divideByConstant(builder, quotient.ref(), sum.ref(), dims.upIters);
  builder.muli32(quotient.ref(), quotient.ref(), dims.upIters);
  builder.subi32(rot, sum.ref(), quotient.ref());
}

// w0 = tx * BN + (wave * CG + cg) * TN. The slot wave * CG + cg is the thread
// id divided by LANES_K.
void SkinnyGEMMEmitter::computeWideBase() {
  if (uniformWide()) {
    builder.muli32(w0.ref(), tile.tx, dims.tileWide);
    return;
  }
  computeLaneWideBase(w0.ref());
  addTileBase(w0.ref());
}

// dst = (wave * CG + cg) * TN, the lane part of w0.
void SkinnyGEMMEmitter::computeLaneWideBase(RegisterRef dst) {
  RegisterRef tid = kernel.workitemId().ref();
  shiftRight(builder, dst, tid, dims.log2LanesK);
  multiplyBy(builder, dst, dst, dims.tn);
}

// dst += tx * BN.
void SkinnyGEMMEmitter::addTileBase(RegisterRef dst) {
  RegisterRange tileBase = sgpr();
  builder.muli32(tileBase.ref(), tile.tx, dims.tileWide);
  builder.addi32(dst, dst, tileBase.ref());
}

// The persistent launch adds the tile's part to the lane part of the wide
// columns every tile: w0 = w0Lane + tx * BN and, for the scale loads, a
// temporary w0Load = w0LoadLane + tx * BN, absent when the wide column is
// uniform and w0 itself serves.
std::optional<RegisterRange>
SkinnyGEMMEmitter::computeTileWideBases(const LaneValues &lane) {
  if (uniformWide()) {
    builder.muli32(w0.ref(), tile.tx, dims.tileWide);
    return std::nullopt;
  }
  RegisterRange tileBase = sgpr();
  builder.muli32(tileBase.ref(), tile.tx, dims.tileWide);
  builder.addi32(w0.ref(), lane.w0Lane->ref(), tileBase.ref());
  RegisterRange w0Load = vgpr();
  builder.addi32(w0Load.ref(), lane.w0Load->ref(), tileBase.ref());
  return w0Load;
}

// The byte address of the lanes that hold scale dwords 0 and 1 of this lane:
// 4 * (j * CG + cg) with the slot j(kl, d) of the MLIR's @scl_slot.
void SkinnyGEMMEmitter::computeAddresses(RegisterRef klv, RegisterRef lane) {
  RegisterRef a0 = addresses.ref(0);
  RegisterRef a1 = addresses.ref(1);
  RegisterRange term = vgpr();
  // The block pair of the lane in its subgroup, times CG.
  bitAnd(builder, a0, klv, dims.subgroupLanes - 1);
  shiftRight(builder, a0, a0, 1);
  shiftLeft(builder, a0, a0, dims.log2ColumnGroups);
  if (dims.lanesK > 8) {
    shiftRight(builder, term.ref(), klv, 3);
    shiftLeft(builder, term.ref(), term.ref(),
              dims.log2Subgroup + dims.log2ColumnGroups);
    builder.addi32(a0, a0, term.ref());
  }
  if (dims.lanesK < 64) {
    shiftRight(builder, term.ref(), lane, dims.log2LanesK);
    builder.addi32(a0, a0, term.ref());
  }
  shiftLeft(builder, a0, a0, 2);
  builder.addi32(a1, a0, dims.halfSubgroup * dims.columnGroups * 4);
}

// Shared scales: the LDS byte offset 4 * (j * R + wave) of the lowest dword of
// this lane, with the slot j(kl, 0) = (kl / 8) * 8 + (kl mod 8) / 2 of LANES_K
// = 64. The other dword is a constant offset away, so one address suffices.
void SkinnyGEMMEmitter::computeSharedAddress(RegisterRef klv, RegisterRef tid) {
  RegisterRef a0 = addresses.ref();
  RegisterRange term = vgpr();
  bitAnd(builder, a0, klv, dims.subgroupLanes - 1);
  shiftRight(builder, a0, a0, 1);
  multiplyBy(builder, a0, a0, 4 * dims.roleSlots);
  shiftRight(builder, term.ref(), klv, 3);
  multiplyBy(builder, term.ref(), term.ref(), 32 * dims.roleSlots);
  builder.addi32(a0, a0, term.ref());
  shiftRight(builder, term.ref(), tid, 6);
  shiftLeft(builder, term.ref(), term.ref(), 2);
  builder.addi32(a0, a0, term.ref());
}

// Subgroup j of S lanes owns the slab j of 64 * S bytes, and the lane reads 16
// bytes at its place in the subgroup.
RegisterRange SkinnyGEMMEmitter::computeLaneBytes(RegisterRef klv) {
  RegisterRange laneBytes = vgpr();
  if (dims.lanesK == dims.subgroupLanes) {
    shiftLeft(builder, laneBytes.ref(), klv, 4);
    return laneBytes;
  }
  RegisterRange inSubgroup = vgpr();
  shiftRight(builder, laneBytes.ref(), klv, dims.log2Subgroup);
  shiftLeft(builder, laneBytes.ref(), laneBytes.ref(), dims.log2Subgroup + 6);
  bitAnd(builder, inSubgroup.ref(), klv, dims.subgroupLanes - 1);
  shiftLeft(builder, inSubgroup.ref(), inSubgroup.ref(), 4);
  builder.addi32(laneBytes.ref(), laneBytes.ref(), inSubgroup.ref());
  return laneBytes;
}

// dst = tid mod R given roleJ = tid / R.
void SkinnyGEMMEmitter::threadInDomain(RegisterRef dst, RegisterRef tid,
                                       RegisterRef roleJ) {
  if (isPowerOfTwo(dims.roleSlots)) {
    bitAnd(builder, dst, tid, dims.roleSlots - 1);
    return;
  }
  builder.muli32(dst, roleJ, dims.roleSlots);
  builder.subi32(dst, tid, dst);
}

// The slot roleJ = id / R of the thread's scale load role, where R = roleSlots
// is not a power of two for some block sizes, so the division is LLVM's
// expansion of the signed division by a constant.
RegisterRange SkinnyGEMMEmitter::computeRoleJ(RegisterRef tid) {
  RegisterRange roleJ = vgpr();
  if (isPowerOfTwo(dims.roleSlots)) {
    shiftRight(builder, roleJ.ref(), tid, log2Exact(dims.roleSlots));
    return roleJ;
  }
  SignedMagic magic = signedMagic(dims.roleSlots);
  RegisterRange sign = vgpr();
  builder.movb32(roleJ.ref(), magic.multiplier);
  builder.create<rocisa::VMulHII32>(roleJ.container(), tid.container(),
                                    roleJ.container());
  if (magic.multiplier < 0)
    builder.addi32(roleJ.ref(), roleJ.ref(), tid);
  if (magic.shift > 0)
    builder.create<rocisa::VAShiftRightI32>(roleJ.container(), magic.shift,
                                            roleJ.container(), "");
  shiftRight(builder, sign.ref(), roleJ.ref(), 31);
  builder.addi32(roleJ.ref(), roleJ.ref(), sign.ref());
  return roleJ;
}

// The byte offset in an iteration of the scale slot roleJ that the lane loads:
// the slab roleJ / S of 64 * S bytes, the block pair roleJ mod (S / 2) of 64
// bytes and the dword step (roleJ mod S) / (S / 2) of 32 * S bytes. roleJ is
// the thread's in shared mode and the lane's cluster otherwise.
RegisterRange SkinnyGEMMEmitter::computeScaleOffset(
    RegisterRef lane, const std::optional<RegisterRange> &sharedRoleJ) {
  std::optional<RegisterRange> roleStorage;
  RegisterRef roleJ = lane;
  if (sharedRoleJ) {
    roleJ = sharedRoleJ->ref();
  } else if (dims.columnGroups > 1) {
    roleStorage = vgpr();
    shiftRight(builder, roleStorage->ref(), lane, dims.log2ColumnGroups);
    roleJ = roleStorage->ref();
  }
  RegisterRange sclOff = vgpr();
  RegisterRange term = vgpr();
  bitAnd(builder, sclOff.ref(), roleJ, dims.halfSubgroup - 1);
  shiftLeft(builder, sclOff.ref(), sclOff.ref(), 6);
  if (dims.lanesK > dims.subgroupLanes) {
    shiftRight(builder, term.ref(), roleJ, dims.log2Subgroup);
    shiftLeft(builder, term.ref(), term.ref(), dims.log2Subgroup + 6);
    builder.addi32(sclOff.ref(), sclOff.ref(), term.ref());
  }
  bitAnd(builder, term.ref(), roleJ, dims.subgroupLanes - 1);
  shiftRight(builder, term.ref(), term.ref(), dims.log2Half);
  shiftLeft(builder, term.ref(), term.ref(), dims.log2Subgroup + 5);
  builder.addi32(sclOff.ref(), sclOff.ref(), term.ref());
  return sclOff;
}

// w0Load = tx * BN + (wave * CG + lane mod CG) * TN: the wide column that the
// lane loads scales for, which differs from the column it owns. In shared mode
// the slot is the thread's row slot tid mod R.
RegisterRange SkinnyGEMMEmitter::computeWideLoadBase(
    RegisterRef tid, RegisterRef lane,
    const std::optional<RegisterRange> &sharedRoleJ) {
  RegisterRange w0Load = uniformWide() ? sgpr() : vgpr();
  if (uniformWide()) {
    builder.muli32(w0Load.ref(), tile.tx, dims.tileWide);
    return w0Load;
  }
  computeWideLoadSlot(w0Load.ref(), tid, lane, sharedRoleJ);
  addTileBase(w0Load.ref());
  return w0Load;
}

// slot = (wave * CG + lane mod CG) * TN, the lane part of w0Load.
void SkinnyGEMMEmitter::computeWideLoadSlot(
    RegisterRef slot, RegisterRef tid, RegisterRef lane,
    const std::optional<RegisterRange> &sharedRoleJ) {
  if (sharedRoleJ) {
    threadInDomain(slot, tid, sharedRoleJ->ref());
  } else {
    if (dims.waves > 1) {
      shiftRight(builder, slot, tid, 6);
      shiftLeft(builder, slot, slot, dims.log2ColumnGroups);
    }
    if (dims.columnGroups > 1) {
      RegisterRange inDomain = vgpr();
      bitAnd(builder, inDomain.ref(), lane, dims.columnGroups - 1);
      if (dims.waves > 1)
        builder.addi32(slot, slot, inDomain.ref());
      else
        builder.movb32(slot, inDomain.ref());
    }
  }
  multiplyBy(builder, slot, slot, dims.tn);
}

SkinnyGEMMEmitter::LaneValues SkinnyGEMMEmitter::computeLaneValues() {
  RegisterRef tid = kernel.workitemId().ref();
  // The block is (64 * waves, 1, 1), so the packed y and z fields of v0 are
  // zero and no mask is needed.
  std::optional<RegisterRange> laneStorage;
  RegisterRef lane = tid;
  if (dims.waves > 1) {
    laneStorage = vgpr();
    bitAnd(builder, laneStorage->ref(), tid, 63);
    lane = laneStorage->ref();
  }
  std::optional<RegisterRange> klStorage;
  RegisterRef klv = lane;
  if (dims.lanesK < 64) {
    klStorage = vgpr();
    bitAnd(builder, klStorage->ref(), tid, dims.lanesK - 1);
    klv = klStorage->ref();
  }
  std::optional<RegisterRange> w0Lane;
  if (dims.grid3d) {
    computeWideBase();
  } else if (!uniformWide()) {
    w0Lane = vgpr();
    computeLaneWideBase(w0Lane->ref());
  }
  if (dims.scalesShared)
    computeSharedAddress(klv, tid);
  else
    computeAddresses(klv, lane);
  RegisterRange laneBytes = computeLaneBytes(klv);
  if (klStorage)
    klStorage->release();
  std::optional<RegisterRange> roleJ;
  if (dims.scalesShared) {
    roleJ = computeRoleJ(tid);
    shiftLeft(builder, ldsStage->ref(), tid, 2);
  }
  RegisterRange sclOff = computeScaleOffset(lane, roleJ);
  std::optional<RegisterRange> w0Load;
  if (dims.grid3d) {
    w0Load = computeWideLoadBase(tid, lane, roleJ);
  } else if (!uniformWide()) {
    w0Load = vgpr();
    computeWideLoadSlot(w0Load->ref(), tid, lane, roleJ);
  }
  // v0 keeps kl for the epilogue; every use of the thread id is done.
  if (dims.waves > 1 || dims.lanesK < 64)
    bitAnd(builder, tid, tid, dims.lanesK - 1);
  return LaneValues{std::move(laneBytes), std::move(sclOff), std::move(w0Load),
                    std::move(w0Lane)};
}

// dBase[r] = (s0 + r < skinnyCols) ? (s0 + r) * ldS + laneBytes : -1, and
// likewise for the wide columns; the validity masks stay alive for the
// stores.
void SkinnyGEMMEmitter::fillDataBases(const SetupScalars &scalars,
                                      LaneValues &lane) {
  constexpr int outOfRange = -1;
  for (int r = 0; r < dims.bm; ++r) {
    RegisterRange col = sgpr();
    builder.addi32(col.ref(), s0.ref(), r);
    builder.cmpi32(Predicate::ult, sValid.ref(2 * r, 2), col.ref(),
                   skinnyCols.ref());
    builder.muli32(col.ref(), col.ref(), scalars.ldS.ref());
    builder.addi32(dBase.ref(r), col.ref(), lane.laneBytes.ref());
    builder.selectb32(dBase.ref(r), sValid.ref(2 * r, 2), dBase.ref(r),
                      outOfRange);
  }
  for (int t = 0; t < dims.tn; ++t) {
    RegisterRef base = dBase.ref(dims.bm + t);
    RegisterRef valid = wValid.ref(2 * t, 2);
    if (uniformWide()) {
      RegisterRange col = sgpr();
      builder.addi32(col.ref(), w0.ref(), t);
      builder.cmpi32(Predicate::ult, valid, col.ref(), wideCols.ref());
      builder.muli32(col.ref(), col.ref(), scalars.ldW.ref());
      builder.addi32(base, col.ref(), lane.laneBytes.ref());
    } else {
      RegisterRange col = vgpr();
      builder.addi32(col.ref(), w0.ref(), t);
      builder.cmpi32(Predicate::ult, valid, col.ref(), wideCols.ref());
      builder.muli32(col.ref(), col.ref(), scalars.ldW.ref());
      builder.addi32(base, col.ref(), lane.laneBytes.ref());
    }
    builder.selectb32(base, valid, base, outOfRange);
  }
  // The persistent launch reuses the lane's bytes every tile.
  if (dims.grid3d)
    lane.laneBytes.release();
}

// sBase[c] is the voffset of the scale dword that the lane loads for column
// c: the load column of the wide operand differs from the lane's own column.
void SkinnyGEMMEmitter::fillScaleBases(const SetupScalars &scalars,
                                       const LaneValues &lane,
                                       RegisterRef w0Load) {
  constexpr int outOfRange = -1;
  if (scaleGroup)
    bitAnd(builder, scaleGroup->ref(), lane.sclOff.ref(), -256);
  for (int r = 0; r < dims.bm; ++r) {
    RegisterRange col = sgpr();
    RegisterRange row = sgpr();
    builder.addi32(col.ref(), s0.ref(), r);
    scaleRowOffset(builder, row.ref(), col.ref(), scalars.rowGroup.ref());
    builder.addi32(sBase.ref(r), row.ref(), lane.sclOff.ref());
    builder.selectb32(sBase.ref(r), sValid.ref(2 * r, 2), sBase.ref(r),
                      outOfRange);
  }
  for (int t = 0; t < dims.tn; ++t) {
    RegisterRef base = sBase.ref(dims.bm + t);
    RegisterRange valid = sgpr(2, 2);
    RegisterRange col = uniformWide() ? sgpr() : vgpr();
    RegisterRange row = uniformWide() ? sgpr() : vgpr();
    builder.addi32(col.ref(), w0Load, t);
    builder.cmpi32(Predicate::ult, valid.ref(), col.ref(), wideCols.ref());
    scaleRowOffset(builder, row.ref(), col.ref(), scalars.rowGroup.ref());
    builder.addi32(base, row.ref(), lane.sclOff.ref());
    builder.selectb32(base, valid.ref(), base, outOfRange);
  }
}

// The selector of the column group of a row: all rows of a 16-row half share
// one when BM or TN divides 16, since s0 and w0 are multiples of them.
RegisterRef SkinnyGEMMEmitter::selector(int column) const {
  bool skinny = column < dims.bm;
  bool skinnyShared = 16 % dims.bm == 0;
  bool wideShared = 16 % dims.tn == 0;
  int skinnyCount = skinnyShared ? 1 : dims.bm;
  if (skinny)
    return selectors.ref(skinnyShared ? 0 : column);
  return selectors.ref(skinnyCount + (wideShared ? 0 : column - dims.bm));
}

// dst = base + ((col >> 4) & 1) * 0x01010101: the byte selector of v_perm_b32
// that packs the two scale dwords of a column whose row is in the lower or
// upper 16 rows of its 32-row group.
void SkinnyGEMMEmitter::selectorValue(RegisterRef dst, RegisterRef col) {
  int selectorBase = dims.subgroupLanes == 8 ? 0x06040200 : 0x06020400;
  if (isScalar(col)) {
    RegisterRange value = sgpr();
    shiftRight(builder, value.ref(), col, 4);
    bitAnd(builder, value.ref(), value.ref(), 1);
    builder.muli32(value.ref(), value.ref(), selectorMaskStep);
    builder.addi32(value.ref(), value.ref(), selectorBase);
    builder.movb32(dst, value.ref());
    return;
  }
  shiftRight(builder, dst, col, 4);
  bitAnd(builder, dst, dst, 1);
  builder.muli32(dst, dst, selectorMaskStep);
  builder.addi32(dst, dst, selectorBase);
}

void SkinnyGEMMEmitter::fillSelectors() {
  if (16 % dims.bm == 0) {
    selectorValue(selector(0), s0.ref());
  } else {
    for (int r = 0; r < dims.bm; ++r) {
      RegisterRange col = sgpr();
      builder.addi32(col.ref(), s0.ref(), r);
      selectorValue(selector(r), col.ref());
    }
  }
  if (16 % dims.tn == 0) {
    selectorValue(selector(dims.bm), w0.ref());
    return;
  }
  for (int t = 0; t < dims.tn; ++t) {
    RegisterRange col = uniformWide() ? sgpr() : vgpr();
    builder.addi32(col.ref(), w0.ref(), t);
    selectorValue(selector(dims.bm + t), col.ref());
  }
}

void SkinnyGEMMEmitter::zeroAccumulators() {
  for (int p = 0; p < dims.accumulators; ++p)
    builder.movb32(acc.ref(p), 0);
}

} // namespace rocm::tensilelite::codegen::detail
