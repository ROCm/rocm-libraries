################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.AsmRegisterPool import RegisterPool
from Tensile.KernelWriter import KernelWriter
from Tensile.Common import globalParameters
import collections
import inspect
from types import SimpleNamespace

class KernelDict(dict):
    def __getattr__(self, name):
        return self[name]

def test_gfx12_compatibility_checks_do_not_apply_to_future_isa():
    kw = KernelWriterAssembly("","")
    kw.version = (13, 0, 0)
    kw.bpeCinternal = 4
    kw.bpr = 4

    assert "s_barrier_signal" not in kw.defineBarrierMacros()
    assert "s_cmpk_eq_u32" in kw.checkIsBetaZero({"ProblemType": {"UseBeta": True}}, "Label", 0)
    assert "gfx12 buffer soffset must be SGPR" not in str(
        kw.chooseGlobalRead(True, 4, 0, "v0", "s[0:3]", 0, 0, "", False))
    assert "gfx12 buffer atomic soffset must be SGPR" not in kw.chooseAtomicCmpswap(
        {"BufferStore": True}, SimpleNamespace(globalOffset=0, addrDVgpr=0), 0, 0, 1, "glc")

def test_gfx9_barrier_assembly_stays_plain_s_barrier():
    kw = KernelWriterAssembly("","")
    kw.version = (9, 0, 0)

    assert kw.syncStr == "s_barrier"
    assert kw.defineBarrierMacros() == ""

def test_gfx9_sync_threads_emits_plain_s_barrier():
    isa = (9, 0, 0)
    oldArchCaps = globalParameters.get("ArchCaps")
    globalParameters["ArchCaps"] = {isa: {"SeparateVscnt": False, "Waitcnt0Disabled": False}}
    kw = KernelWriterAssembly("","")
    kw.version = isa
    kw.kernel = {"WavefrontSize": 32}
    kw.do = {"Sync": True}
    kw.prefetchAcrossPersistent = False
    kernel = KernelDict(
        NumThreads=64,
        ScheduleIterAlg=0,
        PrefetchGlobalRead=0,
        enabledSplitLDS=False,
    )

    try:
        assert kw.syncThreads(kernel, "gfx9 sync") == "s_barrier //gfx9 sync\n"
    finally:
        if oldArchCaps is None:
            del globalParameters["ArchCaps"]
        else:
            globalParameters["ArchCaps"] = oldArchCaps

def test_kernel_writer_direct_barrier_sites_use_syncstr():
    source = inspect.getsource(KernelWriter)

    assert source.count('barrier.addInst(self.syncStr,"")') == 2
    assert "syncCode.addCode(self.syncStr + self.endLine)" in source

def test_gfx12_barrier_assembly_uses_macro():
    kw = KernelWriterAssembly("","")
    kw.version = (12, 0, 1)

    barrierMacro = kw.defineBarrierMacros()

    assert kw.syncStr == "_s_barrier"
    assert "s_barrier_signal" in barrierMacro
    assert "s_barrier_wait" in barrierMacro

def test_gfx12_atomic_cmpswap_modifier_preserves_extra_modifiers():
    kw = KernelWriterAssembly("","")
    kw.version = (12, 0, 1)

    assert kw.gfx12AtomicCmpswapMemoryModifier("glc") == "th:TH_ATOMIC_RT_RETURN"
    assert kw.gfx12AtomicCmpswapMemoryModifier("glc slc") == "th:TH_ATOMIC_RT_RETURN slc"

def test_gfx12_atomic_cmpswap_modifier_rejects_other_architectures():
    kw = KernelWriterAssembly("","")
    kw.version = (13, 0, 0)

    try:
        kw.gfx12AtomicCmpswapMemoryModifier("glc")
    except RuntimeError:
        return

    assert False, "Expected gfx12 atomic modifier helper to reject gfx13"

def test_gfx12_b256_emulation_preserves_outer_module_setup():
    kw = KernelWriterAssembly("","")
    kw.version = (12, 0, 1)
    kw.sgprPool = RegisterPool(8, "s", defaultPreventOverflow=False, printRP=0)
    kw.sgprPool.add(0, 8, "tmp")

    code = str(kw.chooseGlobalRead(True, 32, 0, "v0", "s[0:3]", 0, 0, "", False))

    assert "gfx12 buffer soffset must be SGPR" in code
    assert code.count("_buffer_load_b128") == 2
    assert code.index("gfx12 buffer soffset must be SGPR") < code.index("_buffer_load_b128")

def test_workgroup2_hydration_uses_actual_sgpr_definition():
    kw = KernelWriterAssembly("","")
    kw.sgprs = collections.OrderedDict()

    assert not kw.hasWorkGroup2()
    kw.sgprs["WorkGroup2"] = 4
    assert kw.hasWorkGroup2()

def test_occupancy():
    kw = KernelWriterAssembly("","")
    # numThreads = 256
    assert kw.getOccupancy(256,  10, 65536,   0) == 1
    assert kw.getOccupancy(256,  10, 16384, 128) == 2
    assert kw.getOccupancy(256,  65,  8192,  64) == 3
    assert kw.getOccupancy(256,  10, 65536) == 1
    assert kw.getOccupancy(256,  10, 32768) == 2
    assert kw.getOccupancy(256, 129, 32768) == 1
    assert kw.getOccupancy(256,  10, 16384) == 4
    assert kw.getOccupancy(256, 256, 32768) == 1

    # numThreads = 512
    assert kw.getOccupancy(512,  10, 65536,   0) == 1
    assert kw.getOccupancy(512,  10, 16384, 128) == 1
    assert kw.getOccupancy(512,  65,  8192,  64) == 1
    assert kw.getOccupancy(512,  10, 65536) == 1
    assert kw.getOccupancy(512,  10, 32768) == 2
    assert kw.getOccupancy(512, 129, 32768) == 0
    assert kw.getOccupancy(512,  10, 16384) == 4
    assert kw.getOccupancy(512, 256, 32768) == 0
    assert kw.getOccupancy(512,  13,  1024) == 5

    #----------- unifiedVgprRegs = True ---------------
    # numThreads = 256
    assert kw.getOccupancy(256, 256, 32768, 128, True) ==  1
    assert kw.getOccupancy(256, 128, 65536, 128, True) ==  1
    assert kw.getOccupancy(256, 256, 32768,   0, True) ==  2
    assert kw.getOccupancy(256, 128, 32768, 128, True) ==  2
    assert kw.getOccupancy(256, 128,  1024, 128, True) ==  2
    assert kw.getOccupancy(256,  97,  8192,   0, True) ==  4
    assert kw.getOccupancy(256,  65,  8192,  32, True) ==  4
    assert kw.getOccupancy(256,  64,  8192,  32, True) ==  5
    assert kw.getOccupancy(256,  65,  8192,   0, True) ==  7
    assert kw.getOccupancy(256,   0,  8192,  64, True) ==  8
    assert kw.getOccupancy(256,  24,  8192,  24, True) ==  8
    assert kw.getOccupancy(256,  49,  1024,   0, True) ==  9
    assert kw.getOccupancy(256,  24,  1024,  24, True) == 10
    assert kw.getOccupancy(256,  48,  1024,   0, True) == 10
    assert kw.getOccupancy(256,   3,  1024,  16, True) == 10

    # numThreads = 512
    assert kw.getOccupancy(512, 256, 32768, 128, True) ==  0
    assert kw.getOccupancy(512, 128, 65536, 128, True) ==  1
    assert kw.getOccupancy(512, 256, 32768,   0, True) ==  1
    assert kw.getOccupancy(512, 128, 32768, 128, True) ==  1
    assert kw.getOccupancy(512, 128,  1024, 128, True) ==  1
    assert kw.getOccupancy(512,  97,  8192,   0, True) ==  2
    assert kw.getOccupancy(512,  65,  8192,  32, True) ==  2
    assert kw.getOccupancy(512,  64,  8192,  32, True) ==  2
    assert kw.getOccupancy(512,  65,  8192,   0, True) ==  3
    assert kw.getOccupancy(512,   0,  8192,  64, True) ==  4
    assert kw.getOccupancy(512,  49,  1024,   0, True) ==  4
    assert kw.getOccupancy(512,  24,  1024,  24, True) ==  5
    assert kw.getOccupancy(512,  48,  1024,   0, True) ==  5
    assert kw.getOccupancy(512,   3,  1024,  16, True) ==  5

def test_max_regs():
    kw = KernelWriterAssembly("","")
    assert kw.getMaxRegsForOccupancy(256,  10, 65536,   0) == 256
    assert kw.getMaxRegsForOccupancy(256,  10, 16384, 128) == 128
    assert kw.getMaxRegsForOccupancy(256,  65,  8192,  64) == 84
    assert kw.getMaxRegsForOccupancy(256,  10, 65536) == 256
    assert kw.getMaxRegsForOccupancy(256,  10, 32768) == 128
    assert kw.getMaxRegsForOccupancy(256, 129, 32768) == 256
    assert kw.getMaxRegsForOccupancy(256,  10, 16384) == 64
    assert kw.getMaxRegsForOccupancy(256, 256, 32768) == 256

    # numThreads = 512
    assert kw.getMaxRegsForOccupancy(512,  10, 65536,   0) == 128
    assert kw.getMaxRegsForOccupancy(512,  10, 16384, 128) == 128
    assert kw.getMaxRegsForOccupancy(512,  65,  8192,  64) == 128
    assert kw.getMaxRegsForOccupancy(512,  10, 65536) == 128
    assert kw.getMaxRegsForOccupancy(512,  10, 32768) ==  64
    assert kw.getMaxRegsForOccupancy(512, 129, 32768) == 129
    assert kw.getMaxRegsForOccupancy(512,  10, 16384) ==  32
    assert kw.getMaxRegsForOccupancy(512, 256, 32768) == 256
    assert kw.getMaxRegsForOccupancy(512,  13,  1024) ==  24

    #----------- unifiedVgprRegs = True ---------------
    # numThreads = 256
    assert kw.getMaxRegsForOccupancy(256, 256, 32768, 128, True) ==  256
    assert kw.getMaxRegsForOccupancy(256, 128, 65536, 128, True) ==  256
    assert kw.getMaxRegsForOccupancy(256, 256, 32768,   0, True) ==  256
    assert kw.getMaxRegsForOccupancy(256, 128, 32768, 128, True) ==  128
    assert kw.getMaxRegsForOccupancy(256, 128,  1024, 128, True) ==  128
    assert kw.getMaxRegsForOccupancy(256,  97,  8192,   0, True) ==  128
    assert kw.getMaxRegsForOccupancy(256,  65,  8192,  32, True) ==   96
    assert kw.getMaxRegsForOccupancy(256,  64,  8192,  32, True) ==   64
    assert kw.getMaxRegsForOccupancy(256,  65,  8192,   0, True) ==   72
    assert kw.getMaxRegsForOccupancy(256,   0,  8192,  64, True) ==    0
    assert kw.getMaxRegsForOccupancy(256,  24,  8192,  24, True) ==   40
    assert kw.getMaxRegsForOccupancy(256,  49,  1024,   0, True) ==   56
    assert kw.getMaxRegsForOccupancy(256,  24,  1024,  24, True) ==   24
    assert kw.getMaxRegsForOccupancy(256,  48,  1024,   0, True) ==   48
    assert kw.getMaxRegsForOccupancy(256,   3,  1024,  16, True) ==   32

    # numThreads = 512
    assert kw.getMaxRegsForOccupancy(512, 256, 32768, 128, True) ==  256
    assert kw.getMaxRegsForOccupancy(512, 128, 65536, 128, True) ==  128
    assert kw.getMaxRegsForOccupancy(512, 256, 32768,   0, True) ==  256
    assert kw.getMaxRegsForOccupancy(512, 128, 32768, 128, True) ==  128
    assert kw.getMaxRegsForOccupancy(512, 128,  1024, 128, True) ==  128
    assert kw.getMaxRegsForOccupancy(512,  97,  8192,   0, True) ==  128
    assert kw.getMaxRegsForOccupancy(512,  65,  8192,  32, True) ==   96
    assert kw.getMaxRegsForOccupancy(512,  64,  8192,  32, True) ==   96
    assert kw.getMaxRegsForOccupancy(512,  65,  8192,   0, True) ==   80
    assert kw.getMaxRegsForOccupancy(512,   0,  8192,  64, True) ==    0
    assert kw.getMaxRegsForOccupancy(512,  49,  1024,   0, True) ==   64
    assert kw.getMaxRegsForOccupancy(512,  24,  1024,  24, True) ==   24
    assert kw.getMaxRegsForOccupancy(512,  48,  1024,   0, True) ==   48
    assert kw.getMaxRegsForOccupancy(512,   3,  1024,  16, True) ==   32

# test_occupancy()
# test_max_regs()
