# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

VERSION = "2.01"

# EnqueuesPerSync values (use same value for NumWarmups) - [step, value]
# apply value with N_dim <=step
stepValue_EnqueuesPerSync = [[64*64*8192,200], [256*256*8192,30], [1024*1024*8192,20], [1000000000000000,10]]

dataSize = {'H': 2, 'B': 2, 'S': 4, 'D': 8, 'C': 8, 'Z': 16, 'I8': 1, 'X': 4, 'F8': 1, 'F8N': 1, 'F8B8': 1, 'B8F8': 1, 'X1': 4, 'F4': 1}

LIST_OF_MIN_DIM={'H': 7, 'B': 7, 'S': 3, 'D': 1, 'C': 1, 'Z': 1, 'I8': 7, 'X': 3, 'F8': 7, 'F8N': 7, 'F8B8': 7, 'B8F8': 7, 'X1': 4, 'F4': 7}

depthURange = {}  # [for small/mid MT], [for large  MT]
depthURange['H'] = [[64,128,256,512], [32,64,128,256], [32,64,128], [32,64]]
depthURange['B'] = depthURange['H']
depthURange['S'] = [[16,32,64,128,256], [8,16,32,64,128], [8,16,32,64], [8,16,32]]
depthURange['D'] = [[16,32,64,128], [8,16,32,64], [8,16,32], [8,16]]
depthURange['C'] = depthURange['S']
depthURange['Z'] = depthURange['D']
depthURange['X'] = [[32,64,128,256], [16,32,64,128], [16,32,64], [16,32]]
depthURange['X1'] = depthURange['X']
depthURange['I8'] = [[128,256,512,1024], [64,128,256,512], [64,128,256], [64,128]]
depthURange['F8'] = depthURange['I8']
depthURange['F8N'] = depthURange['I8']
depthURange['F8B8'] = depthURange['F8']
depthURange['B8F8'] = depthURange['F8']
# fp4 MI16x16x128: DepthU must be a multiple of 2*MI_K = 256.
depthURange['F4'] = [[256,512,768,1024], [256,512,768], [256,512], [256]]

computeDataTypeSize = {'H': 4, 'B': 4, 'S': 4, 'D': 8, 'C': 8, 'Z': 16, 'I8': 4, 'X': 4, 'F8': 4, 'F8N': 4, 'F8B8': 4, 'B8F8': 4, 'X1': 4, 'F4': 4}


# TODO update for every new arch, or import from tensilelite commons
validMFMA = {}
validMFMA["H"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16], [32,32,16,1], [16,16,32,1]]
validMFMA["S"] = [[32,32,1,2], [32,32,2,1], [16,16,1,4], [16,16,4,1], [4,4,1,16]]
validMFMA["B"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16], [32,32,16,1], [16,16,32,1]]
validMFMA["D"] = [[16,16,4,1], [4,4,4,4]]
validMFMA["B1k"] = validMFMA["H"]
validMFMA["C"] = validMFMA["S"]
validMFMA["Z"] = validMFMA["D"]
validMFMA["X"] = validMFMA["B"]
validMFMA["X1"] = validMFMA["B"]
validMFMA["F8"] = [[32,32,16,1], [16,16,32,1], [32,32,64,1], [16,16,128,1]]
validMFMA["F4"] = [[16,16,128,1], [32,32,64,1], [32,16,128,1]]
validMFMA["B8"] = validMFMA["F8"]
validMFMA["F8N"] = validMFMA["F8"]
validMFMA["F8B8"] = validMFMA["F8"]
validMFMA["B8F8"] = validMFMA["F8"]
validMFMA["I8_908"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMFMA["I8_940"] = [[32,32,4,2], [32,32,16,1], [16,16,4,4], [16,16,32,1], [4,4,4,16]]
validMFMA["I8"] = validMFMA["H"] + validMFMA["F8"]

MAX_GSU_WORKSPACE_SIZE = 128 * 1024 * 1024

_LARGE_MT0xMT1_DEFAULT = 256 * 464
_REGULAR_MT0xMT1_DEFAULT = 256 * 256


def _build_mt_max_size(large: int, regular: int):
    return {
        'H': large,
        'B': large,
        'S': regular,
        'D': regular,
        'C': 32768,
        'Z': 16384,
        'I8': large,
        'X': large,
        'X1': large,
        'F8': large,
        'F8N': large,
        'F8B8': large,
        'B8F8': large,
        'F4': large,
    }


LIST_OF_MT_MAX_SIZE_DEFAULT = _build_mt_max_size(_LARGE_MT0xMT1_DEFAULT, _REGULAR_MT0xMT1_DEFAULT)


def get_list_of_mt_max_size(search_space=None):
    """Return MT-area cap dict for search_space.

    Stage 1 supports heuristic/generic only, both using default caps.
    """
    return LIST_OF_MT_MAX_SIZE_DEFAULT


# Backward-compatible alias.
LIST_OF_MT_MAX_SIZE = LIST_OF_MT_MAX_SIZE_DEFAULT

ONLY_INCLUDE_MIs_GFX950 = {
    'H':
    [
        #  [16,16,4,4] # never use 16x16x4x4
        #  [32,32,4,2] # never use 32x32x4x2
        [16, 16, 32, 1],
        [32, 32, 16, 1],
    ],
    'B':
    [
        #  [16,16,4,4] # never use 16x16x4x4
        #  [32,32,4,2] # never use 32x32x4x2
        [16, 16, 32, 1],
        [32, 32, 16, 1],
    ],
    'S':
    [
        [16, 16, 4, 1],
        [32, 32, 2, 1],
    ],
    'X':  # For gfx950 we use BF16 MFMAs to implement X(X3)
    [
        [16, 16, 32, 1],
        [32, 32, 16, 1],
    ],
    'X1':  # For gfx950 we use BF16 MFMAs to implement X1
    [
        [16, 16, 32, 1],
        [32, 32, 16, 1],
    ],
    'D':
    [
        [16, 16, 4, 1],
    ],
    'C':
    [
        [16, 16, 4, 1],
    ],
    'Z':
    [
        [16, 16, 4, 1],
    ],
    'I8':
    [
        [32, 32, 16, 1],
        [16, 16, 32, 1],
        [4, 4, 4, 16],
    ],
    'F8':  # similar to I8

    [
        [16, 16, 128, 1],
        [32, 32, 64, 1],
    ],
    'F4':  # fp4 / MX
    [
        [16, 16, 128, 1],
        [32, 32, 64, 1],
    ],
    'F8B8':  # similar to I8
    [
        [16, 16, 128, 1],
        [32, 32, 64, 1],
    ],
    'B8F8':  # similar to I8
    [
        [16, 16, 128, 1],
        [32, 32, 64, 1],
    ],
}

ONLY_INCLUDE_MIs_GFX942 = {
    'H':
    [
        [4, 4, 4, 16],
        #  [16,16,4,4] # never use 16x16x4x4
        [16, 16, 16, 1],
        #  [32,32,4,2] # never use 32x32x4x2
        [32, 32, 8, 1],
    ],

    'B':
    [
        [4, 4, 4, 16],
        #  [16,16,4,4] # never use 16x16x4x4
        [16, 16, 16, 1],
        #  [32,32,4,2] # never use 32x32x4x2
        [32, 32, 8, 1],
    ],
    'S':
    [
        [16, 16, 4, 1],
        [32, 32, 2, 1],
    ],
    'X':
    [
        [16, 16, 32, 1],
        [32, 32, 16, 1],
    ],
    'D':
    [
        [16, 16, 4, 1],
    ],
    'C':
    [
        [16, 16, 4, 1],
    ],
    'Z':
    [
        [16, 16, 4, 1],
    ],
    'I8':
    [
        [32, 32, 16, 1],
        [16, 16, 32, 1],
        [4, 4, 4, 16],
    ],
    'F8':  # similar to I8
    [
        [32, 32, 16, 1],
        [16, 16, 32, 1],

    ],
    'F8N':  # similar to I8
    [
        [32, 32, 16, 1],
        [16, 16, 32, 1],

    ],
    'F8B8':  # similar to I8
    [
        [32, 32, 16, 1],
        [16, 16, 32, 1],
    ],

}

# commenting out other data types so that if and when required it fails and
# we confirm exact MIs needed for each data type.
#
# This is the MI45X baseline, used as-is by gfx1250. The A0 part takes the
# restricted ONLY_INCLUDE_MIs_MI45X_V0 below.
ONLY_INCLUDE_MIs_MI45X = {
    'H': [[16, 16, 32, 1]],
    'B': [[16, 16, 32, 1]],
    'F8': [[16, 16, 128, 1]],
    'F8B8': [[16, 16, 128, 1]],
    'B8F8': [[16, 16, 128, 1]],
    'F4': [[16, 16, 128, 1], [32, 16, 128, 1]],
    # 'X': [[16, 16, 32, 1]],
    # 'X1': [[16, 16, 32, 1]],
    # 'S': [[16, 16, 4, 1]],
    # 'I8': [[16, 16, 32, 1], [16, 16, 64, 1]],
}

from typing import Any, Optional, Tuple

from geko.constants import MX_SCALING_FORMAT, MX_SCALING_FORMAT_PRESWIZZLED, SUPPORTED_ARCH

# Tensile LibraryLogic ``DeviceNames`` as emitted in YAML (asm_full conventions).
LIBRARY_LOGIC_DEVICE_NAMES_GFX950 = '["Device 75a0"]'
LIBRARY_LOGIC_DEVICE_NAMES_GFX942 = '["Device 0049", "Device 0050"]'
LIBRARY_LOGIC_DEVICE_NAMES_GFX1250 = '["Device 73f0"]'
LIBRARY_LOGIC_DEVICE_NAMES_GFX1250_MC = '["Device 75c7"]'

# Shared Tensile LibraryLogic fields (ScheduleName / ArchitectureName / DeviceNames) per silicon family.
_LIBRARY_LOGIC_FIELDS_GFX950 = {
    "ScheduleName": '"gfx950"',
    "ArchitectureName": '"gfx950"',
    "DeviceNames": LIBRARY_LOGIC_DEVICE_NAMES_GFX950,
}
_LIBRARY_LOGIC_FIELDS_GFX942 = {
    "ScheduleName": '"aquavanjaram"',
    "ArchitectureName": '"gfx942"',
    "DeviceNames": LIBRARY_LOGIC_DEVICE_NAMES_GFX942,
}
_LIBRARY_LOGIC_FIELDS_GFX1250 = {
    "ScheduleName": '"gfx1250"',
    "ArchitectureName": '"gfx1250"',
    "DeviceNames": LIBRARY_LOGIC_DEVICE_NAMES_GFX1250,
}
# ScheduleName is the A0/B0 discriminator (gfx1250v0 vs gfx1250) and the ONLY
# one: TensileCreateLibrary (Run.py validLogicFile) drops any logic file whose
# ScheduleName does not match the requested revision, in both directions.
# ArchitectureName stays the real compiler target -- ARCH_COMPILER_TARGET maps
# gfx1250v0 -> gfx1250 for -mcpu.
_LIBRARY_LOGIC_FIELDS_GFX1250_MC = {
    "ScheduleName": '"gfx1250"',
    "ArchitectureName": '"gfx1250"',
    "DeviceNames": LIBRARY_LOGIC_DEVICE_NAMES_GFX1250_MC,
}
# DeviceNames follows the SYSTEM, not the ASIC revision: the un-partitioned
# 256-CU keys take 73f0 and the 96/192-CU (MI450-MC, "Hammer") keys take 75c7,
# on both gfx1250 and gfx1250v0. ScheduleName is the axis that carries the part, which
# is why there are four field sets rather than two.
#
# The rule here is match-the-corpus, not name-the-part. In hipBLASLt's shipped
# logic 73f0 sits on all 431 un-partitioned gfx1250/gfx1250v0 files and 75c7 on
# the 20 files under the four _96cu/_192cu dirs, so this mapping is what merges
# cleanly with it. 73f0 does not identify this silicon: pci.ids calls it Navi 33,
# the same string appears under aldebaran/gfx1201/navi31/navi32/navi33, and an
# MI455X reports 1002:75c1. Nothing downstream notices either way -- Tensile
# only turns DeviceNames into a PciChipId predicate inside
# `if supportsChipIdPredicate(gfxArch)`, which is `return gfx == "gfx950"`
# (Common/Architectures.py), and the runtime gate is the same. Built gfx1250
# libraries contain zero chip-id predicates; gfx950 libraries contain six.
_LIBRARY_LOGIC_FIELDS_GFX1250V0 = {
    "ScheduleName": '"gfx1250v0"',
    "ArchitectureName": '"gfx1250"',
    "DeviceNames": LIBRARY_LOGIC_DEVICE_NAMES_GFX1250,
}
_LIBRARY_LOGIC_FIELDS_GFX1250V0_MC = {
    "ScheduleName": '"gfx1250v0"',
    "ArchitectureName": '"gfx1250"',
    "DeviceNames": LIBRARY_LOGIC_DEVICE_NAMES_GFX1250_MC,
}

# A0 (gfx1250v0) lacks the wide fp4 WMMA, so it takes the baseline minus
# [32, 16, 128, 1]. Spelled out rather than derived by subscript so that
# re-commenting 'F4' above cannot turn this into an import-time KeyError.
ONLY_INCLUDE_MIs_MI45X_V0 = {
    **ONLY_INCLUDE_MIs_MI45X,
    'F4': [[16, 16, 128, 1]],
}

# gfx-style ARCH (YAML) → CUs, XCC, dtype→MI allowlist, Tensile LibraryLogic fields, MX scale value,
# MX block size (keys align with geko.constants.SUPPORTED_ARCH).
# mx_scale: hipblaslt scaleA/scaleB value for MX block scaling (0 = MX not supported on this arch).
# mx_block_size: MXBlockA/MXBlockB size (None = MX not supported on this arch).
_ARCH_SPECS = {
    "gfx950": (256, 8, ONLY_INCLUDE_MIs_GFX950, _LIBRARY_LOGIC_FIELDS_GFX950, 1001, 32),
    "gfx950_128cu": (128, 4, ONLY_INCLUDE_MIs_GFX950, _LIBRARY_LOGIC_FIELDS_GFX950, 1001, 32),
    "gfx942": (304, 8, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942, 0, None),
    "gfx942_80cu": (80, 4, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942, 0, None),
    "gfx942_38cu": (38, 8, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942, 0, None),
    "gfx942_20cu": (20, 4, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942, 0, None),
    "gfx942_228cu": (228, 6, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942, 0, None),
    # MI450-MC ("Hammer") has 6 XCCs, and the partition mode decides how many a
    # single logical device sees:
    #   SPX (_192cu) -- all 6 XCCs drive one device.
    #   DPX (_96cu)  -- the part is split into 2 logical devices of 3 XCCs each,
    #                   so a device sees 3, not 6.
    # The CU count follows the same split, which is why _96cu is exactly half of
    # _192cu on both axes.
    "gfx1250": (256, 8, ONLY_INCLUDE_MIs_MI45X, _LIBRARY_LOGIC_FIELDS_GFX1250, 3, 32),
    "gfx1250_96cu": (96, 3, ONLY_INCLUDE_MIs_MI45X, _LIBRARY_LOGIC_FIELDS_GFX1250_MC, 3, 32),
    "gfx1250_192cu": (192, 6, ONLY_INCLUDE_MIs_MI45X, _LIBRARY_LOGIC_FIELDS_GFX1250_MC, 3, 32),
    "gfx1250v0": (256, 8, ONLY_INCLUDE_MIs_MI45X_V0, _LIBRARY_LOGIC_FIELDS_GFX1250V0, 3, 32),
    "gfx1250v0_96cu": (96, 3, ONLY_INCLUDE_MIs_MI45X_V0, _LIBRARY_LOGIC_FIELDS_GFX1250V0_MC, 3, 32),
    "gfx1250v0_192cu": (192, 6, ONLY_INCLUDE_MIs_MI45X_V0, _LIBRARY_LOGIC_FIELDS_GFX1250V0_MC, 3, 32),
}

HARDWARE_MAP = {
    arch: {
        "CUs": cus,
        "XCC": xcc,
        "ONLY_INCLUDE_MIs": mis,
        "LibraryLogic": ll,
        "mx_scale": mx_scale,
        "mx_block_size": mx_block_size,
    }
    for arch, (cus, xcc, mis, ll, mx_scale, mx_block_size) in _ARCH_SPECS.items()
}

assert set(SUPPORTED_ARCH) == set(_ARCH_SPECS), (
    "SUPPORTED_ARCH must match _ARCH_SPECS / HARDWARE_MAP keys"
)


def mx_format(gemm_config: Any, arch: Optional[str]) -> Optional[Tuple[int, str]]:
    """(block size, scale DataType) of a GemmConfig's MX format on ``arch``.

    None when the GEMM is not block scaled. An unset block is the arch's
    ``mx_block_size`` (32 for an arch outside HARDWARE_MAP), and an unset scale
    type is E8.

    Raises:
        ValueError: If the GEMM is MX and ``arch`` has no MX support.
    """
    if not gemm_config.mx:
        return None
    block = gemm_config.mx_block
    if block is None:
        block = HARDWARE_MAP[arch]["mx_block_size"] if arch in HARDWARE_MAP else 32
        if block is None:
            raise ValueError(f"MX is not supported on ARCH '{arch}'")
    return block, gemm_config.mx_scale_type or "E8"


def mx_scale_code(arch: Optional[str], fmt: Optional[Tuple[int, str]]) -> int:
    """hipBLASLt ``scaleA`` / ``scaleB`` value for an MX format on ``arch``; 0 for None.

    An arch whose ``mx_scale`` is the pre-swizzled value (gfx950) gets it for
    E8 scales on 32-element blocks, the layout its subtile MX kernels read.
    Every other format, and every other arch, gets the canonical value.
    """
    if fmt is None:
        return 0
    if fmt not in MX_SCALING_FORMAT:
        raise ValueError(f"Unsupported MX format: block {fmt[0]}, scale DataType {fmt[1]}")
    if fmt == (32, "E8") and HARDWARE_MAP.get(arch, {}).get("mx_scale") == MX_SCALING_FORMAT_PRESWIZZLED:
        return MX_SCALING_FORMAT_PRESWIZZLED
    return MX_SCALING_FORMAT[fmt]


def mx_format_from_scale_code(code: int) -> Optional[Tuple[int, str]]:
    """(block size, scale DataType) for a hipBLASLt scale value; None when it is not block scaling.

    Raises:
        ValueError: If ``code`` is not a hipblaslt_scaling_format value.
    """
    code = int(code)
    if code in (0, 1, 2):
        return None
    if code == MX_SCALING_FORMAT_PRESWIZZLED:
        return 32, "E8"
    for fmt, value in MX_SCALING_FORMAT.items():
        if value == code:
            return fmt
    raise ValueError(f"Unknown hipBLASLt scale value {code}")

MinKGSU = 256 # preferred

# To design sizes for gridbased library
GRID_BOUNDARY_MT = [256,256] # TODO use large MTs
GRID_BOUNDARY_K_LEVELS = [256,1024,4096,8192,16384]

# Wave configuration
LIST_OF_WAVEs_TO_INCLUDE = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]

# MT Configs
MIN_MT0 = 4
MAX_MT0 = 1024

MIN_MT1 = 4
MAX_MT1 = 1024

# <<< Controls for number of MIs in the config file
# these params are only for MI_FILTER = 2
# tip: lowering this number keeps more MI in the config 
GRANTHRESHOLD = 0.5
GRANTHRESHOLD_128x128 = 0.4 # For MT128x128+, only for MI_FILTER = 2
GRANTHRESHOLD_64x32 = 0.3   # For MT64x32+ to MT128x128, only for MI_FILTER = 2
GRANTHRESHOLD_SMALL = 0.2   # For MT64x32<, only for MI_FILTER = 2
ROUND1 = 2 # larger keeps more MIs
ROUND2 = 3
ROUND3 = 5

# This threshold is used to narrow down the MI selection with LSU. 
# The  larger the number is, the more MI it keeps
LSUTHRESHOLD = 65536

# Kernel cap for heuristic search space (generic uses sys.maxsize).
MAX_NUM_KERNELS_PER_CONFIG = 180_000_000


VALID_BACKENDS = ("ductile", "tensile")
VALID_SEARCH_SPACES = ("heuristic", "generic", "subtile")


# Ductile validation profile: caps elements validated after the last generation.
DUCTILE_VALIDATION_PROFILE_MAP = {
    0: 0, 
    1: 128, 
    2: -1,  # -1 means no cap (use all elements)
}

# Required fields in the input config YAML
REQUIRED_CONFIG_FIELDS = ["TRANSA", "TRANSB", "DataType", "DestDataType", "ComputeDataType", "ARCH"]

# Optional-field defaults per ``ARCH`` (keys = ``SUPPORTED_ARCH``). Built from a
# shared core plus CMS flags: supported on gfx950, not on gfx942-class.
# User YAML overrides via ``setdefault`` in ``load_input_config._prepare_config``.
# To add or change per-ARCH optional defaults, edit ``CONFIG_DEFAULTS_BY_ARCH`` below.
_CONFIG_OPTIONAL_COMMON = {
    "MX": False,
    "StreamK": True,
    # Which hipBLASLt library the tuned logic is destined for.
    #   "Equality" -- exact-size library; the tuned kernel only ever runs on the
    #                 shape it was tuned for, so shape-specialised axes are safe.
    #   "OOB"      -- out-of-box library; a kernel is selected for arbitrary
    #                 sizes, so only shape-agnostic axes may be tuned.
    # Read by the gfx1250 generic profile: it decides whether TDMIterateMode is
    # tunable, and, with StreamK on, whether WGM / WGMXCC / StaggerU / SKXCC are
    # left to origami at run time. Default "OOB" is the conservative one.
    "LIBRARY_TYPE": "OOB",
    # MX block scaling is one setting spelled by three keys (``MX`` above and the
    # two below), resolved into each GemmConfig when the config is loaded:
    #   MX            -- True: block scaling on the arch's ``mx_block_size`` with
    #                    E8 scales. F4 is always MX.
    #   MX_BLOCK      -- block size for A and B (``MXBlockA`` / ``MXBlockB``), 16
    #                    or 32; setting one turns MX on. 0 forces plain non-MX,
    #                    which F4 and ``MX: True`` reject. None leaves it to MX.
    #   MX_SCALE_TYPE -- scale DataType: E8 (default), F8 or E5M3. Emitted as
    #                    ``DataTypeMXSA`` / ``DataTypeMXSB`` only when not E8,
    #                    which is Tensile's default (SolutionStructs/Problem.py).
    # MXFP8 is F8 with MX_BLOCK 32 (``F8BS_MXAE8B32_MXBE8B32``); NVFP4 is F4 with
    # MX_BLOCK 16 and MX_SCALE_TYPE F8 (``MXAF8B16``). Only F4 and F8 take MX.
    "MX_BLOCK": None,
    "MX_SCALE_TYPE": None,
    "search_space": None,
    "MACROTILE_OPT": False,
    "MT_DU": None,
    "USE_HEURISTICS": False,
    "SIZE_OPTION": 0,
    "ONE_SIZE_PER_CONFIG": True,
    "MI_FILTER": 2,
    "EPILOGUES": True,
    "CLUSTER": 0,
    "DUCTILE_VALIDATION_PROFILE": 1,
}

# Config fields that can be overridden by environment variables (if set). Used in _apply_env_config_overrides.
ENV_UPDATABLE_KEYS = {
    "StreamK",
    "MI_FILTER",
    "DUCTILE_VALIDATION_PROFILE",
}

# Input-config keys that ``geko --tune --list`` carries into optim.configure.
# That path reduces the YAML to workload rows and rebuilds the config from
# ARCH / backend / search_space, so these keys, which decide what gets tuned,
# would otherwise fall back to their defaults. The MX format needs no entry:
# each row carries it as its hipBLASLt scaleA / scaleB value, which is decoded
# again when the rows are grouped into GemmConfigs, as for a workload log.
LIST_FORWARDED_KEYS = ("EPILOGUES", "LIBRARY_TYPE")

_CMS_DEFAULTS_GFX950 = {"CMS": True, "CMS_PRIORITY": False}
_CMS_DEFAULTS_GFX942_FAMILY = {"CMS": False, "CMS_PRIORITY": False, "StreamK": False}
_CMS_DEFAULTS_GFX1250 = {"CMS": False, "CMS_PRIORITY": False}

CONFIG_DEFAULTS_BY_ARCH = {
    "gfx950": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX950},
    "gfx950_128cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX950},
    "gfx942": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx942_80cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx942_38cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx942_20cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx942_228cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx1250": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX1250, "StreamK": False},
    "gfx1250_96cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX1250, "StreamK": False},
    "gfx1250_192cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX1250, "StreamK": False},
    "gfx1250v0": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX1250, "StreamK": False},
    "gfx1250v0_96cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX1250, "StreamK": False},
    "gfx1250v0_192cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX1250, "StreamK": False},
}

assert set(CONFIG_DEFAULTS_BY_ARCH) == set(SUPPORTED_ARCH), (
    "CONFIG_DEFAULTS_BY_ARCH keys must match SUPPORTED_ARCH"
)

# Backward-compatible name: full optional defaults for gfx950 (gfx950-class CMS on).
CONFIG_DEFAULTS = CONFIG_DEFAULTS_BY_ARCH["gfx950"]

SEARCH_SPACE_GA_BUDGET = {
    # gfx1250 generic MUST pin the population. Emitting pop_size alone is not
    # enough: Ductile's ``auto_pop_size`` heuristic reads the LARGEST
    # single-variable cardinality and, when that is below pop_size/5, infers the
    # population must be mostly duplicates and halves it -- repeatedly. This space
    # is many small variables (largest ~16 values) with a huge n_perms, so the
    # inference is wrong. Measured: a run left on the default silently went
    # 512 -> 256 -> 128 and evaluated 127 kernels per generation, not 512.
    #
    # An explicit budget makes ConfigSectionGenerator emit ``auto_pop_size: False``
    # next to pop_size, which disables the heuristic. Because Ductile's SearchSpace
    # samples through a validity callback, those 512 are 512 VALID solutions.
    # n_gen 30 matches Ductile's own defaults.yaml (the 20 in the GeneticAlgorithm
    # constructor signature is overridden by that file).
    # Keyed per ARCH, so every gfx1250 revision/partition variant needs an entry --
    # a missing one silently emits no n_gen and no auto_pop_size: False, which is
    # exactly the halving pathology described above.
    **{
        (a, "generic"): {"n_gen": 30, "pop_size": 512}
        for a in (
            "gfx1250",
            "gfx1250_96cu",
            "gfx1250_192cu",
            "gfx1250v0",
            "gfx1250v0_96cu",
            "gfx1250v0_192cu",
        )
    },
}
