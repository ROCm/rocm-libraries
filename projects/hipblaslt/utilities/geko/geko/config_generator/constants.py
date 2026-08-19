# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

VERSION = "2.01"

# EnqueuesPerSync values (use same value for NumWarmups) - [step, value]
# apply value with N_dim <=step
stepValue_EnqueuesPerSync = [[64*64*8192,200], [256*256*8192,30], [1024*1024*8192,20], [1000000000000000,10]]

dataSize = {'H': 2, 'B': 2, 'S': 4, 'D': 8, 'C': 8, 'Z': 16, 'I8': 1, 'X': 4, 'F8': 1,'F8N': 1, 'F8B8': 1, 'B8F8': 1, 'X1': 4}

LIST_OF_MIN_DIM={'H': 7, 'B': 7, 'S': 3, 'D': 1, 'C': 1, 'Z': 1, 'I8': 7, 'X': 3, 'F8': 7,'F8N': 7, 'F8B8': 7, 'B8F8': 7, 'X1': 4}

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

computeDataTypeSize = {'H': 4, 'B': 4, 'S': 4, 'D': 8, 'C': 8, 'Z': 16, 'I8': 4, 'X': 4, 'F8': 4,'F8N': 4, 'F8B8': 4, 'B8F8': 4, 'X1': 4}


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
validMFMA["B8"] = validMFMA["F8"]
validMFMA["F8N"] = validMFMA["F8"]
validMFMA["F8B8"] = validMFMA["F8"]
validMFMA["B8F8"] = validMFMA["F8"]
validMFMA["I8_908"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMFMA["I8_940"] = [[32,32,4,2], [32,32,16,1], [16,16,4,4], [16,16,32,1], [4,4,4,16]]
validMFMA["I8"] = validMFMA["H"] + validMFMA["F8"]

MAX_GSU_WORKSPACE_SIZE = 128 * 1024 * 1024

# TODO: check if we can use larger MT0xMT1 for all datatypes
LARGE_MT0xMT1=256*464
REGULAR_MT0xMT1=256*256
LIST_OF_MT_MAX_SIZE = {
    'H': LARGE_MT0xMT1,
    'B': LARGE_MT0xMT1,
    'S': REGULAR_MT0xMT1, # Is larger MT valid for for F32?
    'D': REGULAR_MT0xMT1,
    'C': 32768,
    'Z': 16384,
    'I8': LARGE_MT0xMT1,
    'X': LARGE_MT0xMT1, # X3 (3 BF16 implementation)
    'X1': LARGE_MT0xMT1, # X1 (1 BF16 implementation)
    'F8': LARGE_MT0xMT1,
    'F8N': LARGE_MT0xMT1,
    'F8B8': LARGE_MT0xMT1,
    'B8F8': LARGE_MT0xMT1}

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

from geko.constants import SUPPORTED_ARCH

# Tensile LibraryLogic ``DeviceNames`` as emitted in YAML (asm_full conventions).
LIBRARY_LOGIC_DEVICE_NAMES_GFX950 = '["Device 75a0"]'
LIBRARY_LOGIC_DEVICE_NAMES_GFX942 = '["Device 0049", "Device 0050"]'

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

# gfx-style ARCH (YAML) → CUs, XCC, dtype→MI allowlist, Tensile LibraryLogic fields
# (keys align with geko.constants.SUPPORTED_ARCH).
_ARCH_SPECS = {
    "gfx950": (256, 8, ONLY_INCLUDE_MIs_GFX950, _LIBRARY_LOGIC_FIELDS_GFX950),
    "gfx950_128cu": (128, 4, ONLY_INCLUDE_MIs_GFX950, _LIBRARY_LOGIC_FIELDS_GFX950),
    "gfx942": (304, 8, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942),
    "gfx942_80cu": (80, 4, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942),
    "gfx942_38cu": (38, 8, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942),
    "gfx942_20cu": (20, 4, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942),
    "gfx942_228cu": (228, 6, ONLY_INCLUDE_MIs_GFX942, _LIBRARY_LOGIC_FIELDS_GFX942),
}

HARDWARE_MAP = {
    arch: {
        "CUs": cus,
        "XCC": xcc,
        "ONLY_INCLUDE_MIs": mis,
        "LibraryLogic": ll,
    }
    for arch, (cus, xcc, mis, ll) in _ARCH_SPECS.items()
}

assert set(SUPPORTED_ARCH) == set(_ARCH_SPECS), (
    "SUPPORTED_ARCH must match _ARCH_SPECS / HARDWARE_MAP keys"
)

MinKGSU = 256 # preferred

# To design sizes for gridbased library
GRID_BOUNDARY_MT = [256,256] # TODO use large MTs
GRID_BOUNDARY_K_LEVELS = [256,1024,4096,8192,16384]

# Wave configuration
LIST_OF_WAVEs_TO_INCLUDE = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]

# MT Configs
MIN_MT0 = 4
MAX_MT0 = 512

MIN_MT1 = 4
MAX_MT1 = 512

MAX_MT_AREA = 464 * 256

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

# Cap on kernels per config file for non-GA (exhaustive) tuning.
# In GA mode this is ignored (set to sys.maxsize). User can override via config YAML.
MAX_NUM_KERNELS_PER_CONFIG = 180_000_000


# GA VALIDATION PROFILE NUM ELEMENTS TO VALIDATE
# This is used to cap the number of elements used for GA kernel validation 
# after the last generation.
GA_VALIDATION_PROFILE_MAP = {
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
    "StreamK": True,
    "GA": True,
    "MACROTILE_OPT": False,
    "MT_DU": None,
    "USE_HEURISTICS": False,
    "SIZE_OPTION": 0,
    "ONE_SIZE_PER_CONFIG": True,
    "MI_FILTER": 2,
    "EPILOGUES": True,
    "CLUSTER": 0,
    "GA_VALIDATION_PROFILE": 1,
}

# Config fields that can be overridden by environment variables (if set). Used in _apply_env_config_overrides.
ENV_UPDATABLE_KEYS = {
    "StreamK",
    "MI_FILTER",
    "GA_VALIDATION_PROFILE",
}

_CMS_DEFAULTS_GFX950 = {"CMS": True, "CMS_PRIORITY": False}
_CMS_DEFAULTS_GFX942_FAMILY = {"CMS": False, "CMS_PRIORITY": False}

CONFIG_DEFAULTS_BY_ARCH = {
    "gfx950": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX950},
    "gfx950_128cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX950},
    "gfx942": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx942_80cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx942_38cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx942_20cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
    "gfx942_228cu": {**_CONFIG_OPTIONAL_COMMON, **_CMS_DEFAULTS_GFX942_FAMILY},
}

assert set(CONFIG_DEFAULTS_BY_ARCH) == set(SUPPORTED_ARCH), (
    "CONFIG_DEFAULTS_BY_ARCH keys must match SUPPORTED_ARCH"
)

# Backward-compatible name: full optional defaults for gfx950 (gfx950-class CMS on).
CONFIG_DEFAULTS = CONFIG_DEFAULTS_BY_ARCH["gfx950"]
