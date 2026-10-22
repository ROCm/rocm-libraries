// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/Kernel.hpp>
#include <tensilelite/codegen/SkinnyGEMM.hpp>

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <memory>
#include <string>

namespace rocm::tensilelite::codegen {

namespace {

// Matches #red_row and the gpu.block_dims of the MLIR.
constexpr int reduceBlock = 256;

struct ArgumentOffsets {
  int d;
  int ws;
  int ldD;
  int m;
  int n;
  int splits;
};

ArgumentOffsets declareArguments(KernelSignature &signature) {
  ArgumentOffsets offsets;
  offsets.d = signature.addPointer("D");
  offsets.ws = signature.addPointer("ws");
  offsets.ldD = signature.addScalar("ldD", 4);
  offsets.m = signature.addScalar("m", 4);
  offsets.n = signature.addScalar("n", 4);
  offsets.splits = signature.addScalar("splits", 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeY, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeZ, 4);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountY, 4);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountZ, 4);
  return offsets;
}

// Emits @splitk_reduce of the MLIR reference, one method per part of it.
class SplitKReduceEmitter {
public:
  SplitKReduceEmitter(Kernel &kernel, const ArgumentOffsets &offsets)
      : kernel(kernel), builder(kernel.builder()), offsets(offsets),
        splits(builder.allocateSgpr(1)), n(builder.allocateSgpr(1)),
        m(builder.allocateSgpr(1)), ldD(builder.allocateSgpr(1)),
        wsRsrc(builder.allocateSgpr(4, 4)), dRsrc(builder.allocateSgpr(4, 4)),
        tmp(builder.allocateSgpr(1)), rowBytes(builder.allocateSgpr(1)),
        sliceBytes(builder.allocateSgpr(1)), split(builder.allocateSgpr(1)),
        savedExec(builder.allocateSgpr(2, 2)), row(builder.allocateVgpr(1)),
        base(builder.allocateVgpr(1)), value(builder.allocateVgpr(1)) {}

  void emit() {
    loadArguments();
    computeRows();
    makeResources();
    sumSplits();
    storeBf16();
  }

private:
  // The pointers load straight into words 0 and 1 of their descriptors.
  void loadArguments() {
    kernel.loadArgument(splits.ref(), offsets.splits);
    kernel.loadArgument(n.ref(), offsets.n);
    kernel.loadArgument(m.ref(), offsets.m);
    kernel.loadArgument(ldD.ref(), offsets.ldD);
    kernel.loadArgument(wsRsrc.ref(0, 2), offsets.ws);
    kernel.loadArgument(dRsrc.ref(0, 2), offsets.d);
  }

  // Computes i, inRange (in vcc) and iLoad, which overwrites i.
  void computeRows() {
    // The block is (256, 1, 1), so the packed y and z fields of v0 are zero and
    // no mask is needed.
    RegisterRef tid = kernel.workitemId().ref();
    builder.create<rocisa::SLShiftLeftB32>(tmp.container(), 8,
                                           kernel.workgroupIdX().container());
    builder.addi32(row.ref(), tmp, tid);
    // Scalar loads return out of order, so only lgkmcnt(0) guarantees any of
    // the six arguments.
    builder.createWaitCnt(WaitCounts{.scalarMemory = 0});
    builder.create<rocisa::VCmpGtU32>(std::make_shared<rocisa::VCC>(),
                                      m.container(), row.container());
    builder.subi32(tmp.ref(), m, 1);
    // Both operands are below 2^31 because m <= 2^20 and i < m + 256, so the
    // signed min equals the unsigned one; rocisa has no v_min_u32.
    builder.create<rocisa::VMinI32>(row.container(), tmp.container(),
                                    row.container());
  }

  // Descriptors of ws (wsBytes) and D (dBytes), which bound the accesses.
  void makeResources() {
    builder.create<rocisa::SLShiftLeftB32>(rowBytes.container(), 2,
                                           m.container());
    builder.muli32(sliceBytes.ref(), n, rowBytes);
    builder.muli32(wsRsrc.ref(2), splits, sliceBytes);
    // dBytes = ((n - 1) * ldD + m) * 2, as (2n - 2) * ldD + 2m.
    RegisterRef dBytes = dRsrc.ref(2);
    builder.create<rocisa::SLShiftLeftB32>(dBytes.container(), 1,
                                           n.container());
    builder.addi32(dBytes, dBytes, -2);
    builder.muli32(dBytes, dBytes, ldD);
    builder.create<rocisa::SLShiftLeftB32>(tmp.container(), 1, m.container());
    builder.addi32(dBytes, dBytes, tmp);
    builder.makeRsrc(wsRsrc.ref(), wsRsrc.ref(0, 2), wsRsrc.ref(2));
    builder.makeRsrc(dRsrc.ref(), dRsrc.ref(0, 2), dBytes);
  }

  // sum = the f32 sum over the splits. Each lane loads its element of ws at
  // (z * m * n + j * m + iLoad) * 4, so splits >= 1 is a precondition.
  void sumSplits() {
    // The row register is dead after base, so it holds the sum.
    RegisterRef sum = row.ref();
    builder.create<rocisa::VLShiftLeftB32>(row.container(), 2, row.container());
    builder.muli32(tmp.ref(), kernel.workgroupIdY(), rowBytes);
    builder.addi32(base.ref(), tmp, row);
    builder.movb32(split.ref(), 0);
    builder.movb32(sum, 0);
    std::shared_ptr<rocisa::Label> loop = builder.createLabel("loop");
    builder.insert(loop);
    builder.muli32(tmp.ref(), split, sliceBytes);
    builder.addi32(value.ref(), tmp, base);
    builder.create<rocisa::BufferLoadB32>(value.container(), value.container(),
                                          wsRsrc.container(), 0,
                                          rocisa::MUBUFModifiers(true, 0));
    // The load is the newest vector memory operation, so nothing is issued
    // after it and vmcnt waits down to 0.
    int loadsIssuedAfter = 0;
    builder.createWaitCnt(WaitCounts{.vectorLoads = loadsIssuedAfter});
    builder.create<rocisa::VAddF32>(sum.container(), sum.container(),
                                    value.container());
    builder.addi32(split.ref(), split, 1);
    builder.create<rocisa::SCmpLtI32>(split.container(), splits.container());
    builder.create<rocisa::SCBranchSCC1>(loop->getLabelName());
  }

  // Truncates the sum to bf16 and stores it where inRange.
  void storeBf16() {
    RegisterRef sum = row.ref();
    RegisterRef tid = kernel.workitemId().ref();
    RegisterRef blockX = kernel.workgroupIdX().ref();
    // truncf keeps the top 16 bits instead of rounding to nearest.
    builder.create<rocisa::VLShiftRightB32>(sum.container(), 16,
                                            sum.container());
    std::shared_ptr<rocisa::Label> skip = builder.createLabel("skip");
    builder.create<rocisa::SAndSaveExecB64>(savedExec.container(),
                                            std::make_shared<rocisa::VCC>());
    builder.create<rocisa::SCBranchVCCZ>(skip->getLabelName());
    // dOff = (j * ldD + i) * 2, recomputed from the thread id; the load
    // address register is dead.
    RegisterRef dOff = value.ref();
    builder.create<rocisa::VLShiftLeftB32>(dOff.container(), 1,
                                           tid.container());
    builder.create<rocisa::SLShiftLeftB32>(tmp.container(), 9,
                                           blockX.container());
    builder.addi32(dOff, tmp, dOff);
    builder.muli32(tmp.ref(), kernel.workgroupIdY(), ldD);
    builder.create<rocisa::SLShiftLeftB32>(tmp.container(), 1, tmp.container());
    builder.addi32(dOff, tmp, dOff);
    builder.create<rocisa::BufferStoreB16>(sum.container(), dOff.container(),
                                           dRsrc.container(), 0,
                                           rocisa::MUBUFModifiers(true, 0));
    builder.insert(skip);
    builder.create<rocisa::SMovB64>(std::make_shared<rocisa::EXEC>(),
                                    savedExec.container());
    builder.createEndProgram();
  }

  Kernel &kernel;
  IRBuilder &builder;
  ArgumentOffsets offsets;
  RegisterRange splits, n, m, ldD, wsRsrc, dRsrc, tmp, rowBytes, sliceBytes,
      split, savedExec, row, base, value;
};

} // namespace

std::string emitSplitKReduce(const Target &target) {
  Kernel kernel(target, "splitk_reduce", reduceBlock);
  // The workgroup id y is the column j.
  kernel.enableWorkgroupIds(true, false);
  ArgumentOffsets offsets = declareArguments(kernel.signature());
  SplitKReduceEmitter(kernel, offsets).emit();
  return kernel.toAssembly();
}

} // namespace rocm::tensilelite::codegen
