import subprocess
from functools import lru_cache
from typing import Tuple

from .Architectures import isaToGfx


def _tryAssembler(
    isaVersion: Tuple[int, int, int],
    assemblerPath: str,
    asmString: str,
    debug: bool = False,
    *options
) -> bool:
    """
    Try to assemble the asmString for the specified target processor
    Success is defined as assembler returning no error code or stderr/stdout
    """
    options = list(options)

    if isaVersion[0] >= 10:
        options += ["-mwavefrontsize64"]

    args = [
        str(assemblerPath),
        "-x",
        "assembler",
        "-target",
        "amdgcn-amdhsa",
        "-mcpu=" + isaToGfx(isaVersion),
        *options,
        "-",
    ]

    result = subprocess.run(
        args, input=asmString.encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output = result.stdout.decode()

    if debug:
        print("isaVersion: ", isaVersion)
        print("asm_cmd:", " ".join(args))
        print("asmString: ", asmString)
        print("output: ", output)
        print("return code: ", result.returncode)

    if output != "" or result.returncode != 0:
        return False
    return True


########################################
# Get Caps
########################################


@lru_cache()
def initAsmCaps(isaVersion, assemblerPath, isDebug) -> dict:
    """Determine assembler capabilities by testing short instructions sequences"""
    rv = {}
    # fmt: off
    rv["SupportedISA"]      = _tryAssembler(isaVersion, assemblerPath, "", isDebug)
    rv["HasExplicitCO"]     = _tryAssembler(isaVersion, assemblerPath, "v_add_co_u32 v0,vcc,v0,1", isDebug)
    rv["HasExplicitNC"]     = _tryAssembler(isaVersion, assemblerPath, "v_add_nc_u32 v0,v0,1", isDebug)

    rv["HasDirectToLds"]    = _tryAssembler(isaVersion, assemblerPath, "buffer_load_dword v36, s[24:27], s28 offen offset:0 lds", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "buffer_load_b32 v36, s[24:27], s28 offen offset:0 lds", isDebug)
    rv["HasDirectToLdsx4"]  = _tryAssembler(isaVersion, assemblerPath, "buffer_load_dwordx4 v36, s[24:27], s28 offen offset:0 lds", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "buffer_load_b128 v36, s[24:27], s28 offen offset:0 lds", isDebug)
    rv["HasAddLshl"]        = _tryAssembler(isaVersion, assemblerPath, "v_add_lshl_u32 v47, v36, v34, 0x2", isDebug)
    rv["HasLshlOr"]         = _tryAssembler(isaVersion, assemblerPath, "v_lshl_or_b32 v47, v36, 0x2, v34", isDebug)
    rv["HasSMulHi"]         = _tryAssembler(isaVersion, assemblerPath, "s_mul_hi_u32 s47, s36, s34", isDebug)

    rv["HasMFMA_explictB"]  = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31]", isDebug)
    rv["HasMFMA"]           = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]", isDebug) or rv["HasMFMA_explictB"]
    rv["HasMFMA_f64"]       = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f64_16x16x4f64 v[0:7], v[32:33], v[36:37], v[0:7]", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "v_mfma_f64_16x16x4_f64 v[0:7], v[32:33], v[36:37], v[0:7]", isDebug)
    rv["HasMFMA_bf16_1k"]   = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_32x32x4bf16_1k a[0:31], v[32:33], v[36:37], a[0:31]", isDebug)
    rv["HasMFMA_f8"]        = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]", isDebug)
    rv["HasMFMA_b8"]        = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]", isDebug)
    rv["HasMFMA_f8f6f4"]    = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3]", isDebug)

    rv["HasMFMA_xf32"]      = _tryAssembler(isaVersion, assemblerPath, "v_mfma_f32_32x32x4_xf32 a[0:15], v[32:33], v[36:37], a[0:15]", isDebug)
    rv["HasSMFMA"]          = _tryAssembler(isaVersion, assemblerPath, "v_smfmac_f32_32x32x16_f16 a[0:15], v[32:33], v[36:39], v[40]", isDebug)
    rv["HasWMMA"]           = _tryAssembler(isaVersion, assemblerPath, "v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]", isDebug)
    rv["HasWMMA_V1"]        = _tryAssembler(isaVersion, assemblerPath, "v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]", isDebug)
    rv["HasWMMA_V2"]        = _tryAssembler(isaVersion, assemblerPath, "v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]", isDebug)

    rv["v_mac_f16"]         = _tryAssembler(isaVersion, assemblerPath, "v_mac_f16 v47, v36, v34", isDebug)

    rv["v_fma_f16"]         = _tryAssembler(isaVersion, assemblerPath, "v_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0,0]", isDebug)
    rv["v_fmac_f16"]        = _tryAssembler(isaVersion, assemblerPath, "v_fma_f16 v47, v36, v34", isDebug)

    rv["v_pk_fma_f16"]      = _tryAssembler(isaVersion, assemblerPath, "v_pk_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0]", isDebug)
    rv["v_pk_fmac_f16"]     = _tryAssembler(isaVersion, assemblerPath, "v_pk_fma_f16 v47, v36, v34", isDebug)

    rv["v_pk_add_f32"]      = _tryAssembler(isaVersion, assemblerPath, "v_pk_add_f32 v[48:49], v[36:37], v[0:1]", isDebug)
    rv["v_pk_mul_f32"]      = _tryAssembler(isaVersion, assemblerPath, "v_pk_mul_f32 v[48:49], v[36:37], v[0:1]", isDebug)

    rv["v_pk_mul_f32"]      = _tryAssembler(isaVersion, assemblerPath, "v_pk_mul_f32 v[20:21], v[18:19], v[20:21]", isDebug)

    rv["v_mad_mix_f32"]     = _tryAssembler(isaVersion, assemblerPath, "v_mad_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]", isDebug)
    rv["v_fma_mix_f32"]     = _tryAssembler(isaVersion, assemblerPath, "v_fma_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]", isDebug)

    rv["v_dot2_f32_f16"]    = _tryAssembler(isaVersion, assemblerPath, "v_dot2_f32_f16 v20, v36, v34, v20", isDebug)
    rv["v_dot2c_f32_f16"]   = _tryAssembler(isaVersion, assemblerPath, "v_dot2c_f32_f16 v47, v36, v34", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "v_dot2acc_f32_f16 v47, v36, v34", isDebug)

    rv["v_dot4_i32_i8"]     = _tryAssembler(isaVersion, assemblerPath, "v_dot4_i32_i8 v47, v36, v34", isDebug)
    rv["v_dot4c_i32_i8"]    = _tryAssembler(isaVersion, assemblerPath, "v_dot4c_i32_i8 v47, v36, v34", isDebug)
    rv["VOP3v_dot4_i32_i8"] = _tryAssembler(isaVersion, assemblerPath, "v_dot4_i32_i8 v47, v36, v34, v47", isDebug)

    rv["v_mac_f32"]         = _tryAssembler(isaVersion, assemblerPath, "v_mac_f32 v20, v21, v22", isDebug)
    rv["v_fma_f32"]         = _tryAssembler(isaVersion, assemblerPath, "v_fma_f32 v20, v21, v22, v23", isDebug)
    rv["v_fmac_f32"]        = _tryAssembler(isaVersion, assemblerPath, "v_fmac_f32 v20, v21, v22", isDebug)

    rv["v_fma_f64"]         = _tryAssembler(isaVersion, assemblerPath, "v_fma_f64 v[20:21], v[22:23], v[24:25], v[20:21]", isDebug)

    rv["v_mov_b64"]         = _tryAssembler(isaVersion, assemblerPath, "v_mov_b64 v[0:1], v[2:3]", isDebug)

    rv["HasBF16CVT"]        = _tryAssembler(isaVersion, assemblerPath, "v_cvt_f32_bf16 v0, v1", isDebug)
    rv["Hascvtfp8_f16"]     = _tryAssembler(isaVersion, assemblerPath, "v_cvt_scalef32_pk_fp8_f16 v[0], v[1], 0 op_sel:[0,0,0,0]", isDebug)
    rv["Hascvtf16_fp8"]     = _tryAssembler(isaVersion, assemblerPath, "v_cvt_scalef32_f16_fp8 v[0], v[1], 0 op_sel:[0,0,0,0]", isDebug)
    
    rv["HasLDSTr"]          = _tryAssembler(isaVersion, assemblerPath,  "ds_read_b64_tr_b16 v[0:1], v0 offset: 0", isDebug)

    rv["v_prng_b32"]        = _tryAssembler(isaVersion, assemblerPath, "v_prng_b32 v47, v36", isDebug)

    rv["HasAtomicAdd"]      = _tryAssembler(isaVersion, assemblerPath, "buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen offset:0", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "buffer_atomic_add_f32 v0, v1, s[0:3], null offen offset:0", isDebug)
    rv["HasGLCModifier"]    = _tryAssembler(isaVersion, assemblerPath, "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, glc", isDebug)
    rv["HasMUBUFConst"]     = _tryAssembler(isaVersion, assemblerPath, "buffer_load_dword v40, v36, s[24:27], 1 offen offset:0", isDebug) \
                                or _tryAssembler(isaVersion, assemblerPath, "buffer_load_b32 v40, v36, s[24:27], 1 offen offset:0", isDebug)
    rv["HasSCMPK"]          = _tryAssembler(isaVersion, assemblerPath, "s_cmpk_gt_u32 s56, 0x0", isDebug)

    rv["HasGLCModifier"]    = _tryAssembler(isaVersion, assemblerPath, "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc", isDebug)

    rv["HasNTModifier"]     = _tryAssembler(isaVersion, assemblerPath, "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, nt", isDebug)

    rv["HasNewBarrier"]     = _tryAssembler(isaVersion, assemblerPath, "s_barrier_wait -1", isDebug)
    # fmt: on

    if _tryAssembler(isaVersion, assemblerPath, "s_waitcnt vmcnt(63)", isDebug):
        rv["MaxVmcnt"] = 63
    elif _tryAssembler(isaVersion, assemblerPath, "s_waitcnt vmcnt(15)", isDebug):
        rv["MaxVmcnt"] = 15
    else:
        rv["MaxVmcnt"] = 0

    # TODO- Need to query the max cap, just like vmcnt as well?
    rv["MaxLgkmcnt"] = 15

    rv["SupportedSource"] = True

    return rv


@lru_cache()
def initArchCaps(isaVersion) -> dict:
    rv = {}
    # fmt: off
    rv["HasEccHalf"]         = (isaVersion in [(9,0,6), (9,0,8), (9,0,10), (9,4,0), (9,4,1), (9,4,2), (9,5,0)])
    rv["Waitcnt0Disabled"]   = (isaVersion in [(9,0,8), (9,0,10), (9,4,0), (9,4,1), (9,4,2), (9,5,0)])
    rv["HasLDSGT64K"]        = (isaVersion in [(9,5,0)])
    rv["SeparateVscnt"]      = isaVersion[0] in (10, 11)
    rv["SeparateLGKMcnt"]    = isaVersion[0] == (12)
    rv["SeparateVMcnt"]      = isaVersion[0] == (12)
    rv["CMPXWritesSGPR"]     = isaVersion[0] not in (10, 11, 12)
    rv["HasWave32"]          = isaVersion[0] in (10, 11, 12)
    rv["HasAccCD"]           = (isaVersion in [(9,0,10), (9,4,0), (9,4,1), (9,4,2), (9,5,0)])
    rv["ArchAccUnifiedRegs"] = (isaVersion in [(9,0,10), (9,4,0), (9,4,1), (9,4,2), (9,5,0)])
    rv["CrosslaneWait"]      = (isaVersion in [(9,4,0), (9,4,1), (9,4,2), (9,5,0)])
    rv["ForceStoreSC1"]      = (isaVersion in [(9,4,0), (9,4,1)])
    rv["TransOpWait"]        = (isaVersion in [(9,4,0), (9,4,1), (9,4,2), (9,5,0)])
    rv["SDWAWait"]           = (isaVersion in [(9,4,0), (9,4,1), (9,4,2), (9,5,0)])
    rv["VgprBank"]           = (isaVersion[0] in (10, 11, 12))
    rv["DSLow16NotPreserve"]       = isaVersion[0] == (12)
    rv["WrokGroupIdFromTTM"] = isaVersion[0] == (12)
    rv["NoSDWA"]             = isaVersion[0] == (12)
    rv["VOP3ByteSel"]      = isaVersion[0] == (12)
    rv["HasFP8_OCP"]         = isaVersion[0] == (12)
    # fmt: on
    return rv


def initRegisterCaps(isaVersion, archCaps) -> dict:
    rv = {}
    rv["MaxVgpr"] = 256
    # max allowed is 112 out of 112 , 6 is used by hardware 4 SGPRs are wasted
    rv["MaxSgpr"] = 102

    rv["PhysicalMaxVgpr"] = 512
    rv["PhysicalMaxSgpr"] = 800

    if isaVersion[0] == 10:
        rv["PhysicalMaxVgprCU"] = 1024 * 32
    elif isaVersion[0] == 11:
        if isaVersion[2] == 2:
            rv["PhysicalMaxVgprCU"] = 1024 * 32
        else:
            rv["PhysicalMaxVgprCU"] = 1536 * 32
    elif isaVersion[0] == 12:
        rv["PhysicalMaxVgprCU"] = 1536 * 32
    elif isaVersion[0] == 9:
        if archCaps["ArchAccUnifiedRegs"]:
            rv["PhysicalMaxVgprCU"] = 2048 * 64
        else:
            rv["PhysicalMaxVgprCU"] = 1024 * 64
    elif isaVersion[0] == 8:
        rv["PhysicalMaxVgprCU"] = 1024 * 64
    elif isaVersion[0] == 0:
        rv["PhysicalMaxVgprCU"] = 0
    else:
        assert 0, "No valid VGPR value for this platform"

    return rv


def initAsmBugs(asmCaps) -> dict:
    rv = {}
    rv["ExplicitCO"] = asmCaps["HasExplicitCO"]
    rv["ExplicitNC"] = asmCaps["HasExplicitNC"]

    return rv
