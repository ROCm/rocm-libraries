################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import functools
import math
import os
import sys
import time
import re

from inspect import currentframe, getframeinfo
from copy import deepcopy
from enum import Enum
from math import log
from pathlib import Path
from typing import Sequence, Tuple, Optional

from Tensile import __version__

from rocisa import rocIsa

import pickle

def fastdeepcopy(x):
    # Note: Some object can't be pickled
    return pickle.loads(pickle.dumps(x))

def isSubtileMultiDU(kernel) -> bool:
    """True when a subtile kernel runs in multi-DU mode.

    Multi-DU means a data tensor's per-uid DepthU (_DepthUA/_DepthUB) is
    smaller than the loop DepthU, i.e. the unroll is split into sub-iterations
    (currently the MXFP8 swizzle path). Single helper so the detection is not
    re-derived inline across the codegen (AsmStoreState, GlobalWriteBatch,
    KernelWriterAssembly).
    """
    du = kernel["DepthU"]
    return kernel.get("_DepthUA", du) < du or kernel.get("_DepthUB", du) < du

# Global
_global_ti = rocIsa.getInstance()

_verbosity = 1

def setVerbosity(v: int):
    global _verbosity
    _verbosity = v

def getVerbosity():
    return _verbosity

################################################################################
# Printing
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints
################################################################################
def print1(message):
    if getVerbosity() >= 1:
        print(message)
        sys.stdout.flush()


def print2(message):
    if getVerbosity() >= 2:
        print(message)
        sys.stdout.flush()


def printWarning(message):
    print("Tensile::WARNING: %s" % message)
    sys.stdout.flush()


def printExit(message):
    print("Tensile::FATAL: %s" % message)
    sys.stdout.flush()
    sys.exit(-1)

# get param values from structures.
def hasParam(name, structure):
    if isinstance(structure, list):
        for l in structure:
            if hasParam(name, l):
                return True
        return False
    elif isinstance(structure, dict):
        return name in structure
    else:
        return name == structure


def isExe(filePath):
    return os.path.isfile(filePath) and os.access(filePath, os.X_OK)


def locateExe(defaultPath, exeName):  # /opt/rocm/bin, hip-clang
    # look in defaultPath first
    if defaultPath:
        exePath = os.path.join(defaultPath, exeName)
        if isExe(exePath):
            return exePath
    # look in PATH second
    for path in os.environ["PATH"].split(os.pathsep):
        exePath = os.path.join(path, exeName)
        if isExe(exePath):
            return exePath

    raise OSError(f"Failed to locate {exeName}")


def ensurePath(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except OSError:
        raise OSError('Failed to create directory "%s" ' % (path))
    return path


def roundUp(f):
    return (int)(math.ceil(f))


def elineno():
    """
    Return the file name and line number of the caller.
    """
    frame = getframeinfo(currentframe().f_back)
    return f"{Path(frame.filename).name}:{frame.lineno}"


################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
def versionIsCompatible(queryVersionString):
    (qMajor, qMinor, qStep) = queryVersionString.split(".")
    (tMajor, tMinor, tStep) = __version__.split(".")

    # major version must match exactly
    if qMajor != tMajor:
        return False

    # minor.patch version must be >=
    if int(qMinor) > int(tMinor):
        return False
    if qMinor == tMinor:
        if int(qStep) > int(tStep):
            return False
    return True


################################################################################
# Progress Bar Printing
# prints "||||" up to width
################################################################################
class ProgressBar:
    def __init__(self, maxValue, width=80):
        self.char = "|"
        self.maxValue = maxValue
        self.width = width
        self.maxTicks = self.width - 7

        self.priorValue = 0
        self.fraction = 0
        self.numTicks = 0
        self.createTime = time.time()

    def increment(self, value=1):
        self.update(self.priorValue + value)

    def update(self, value):
        currentFraction = 1.0 * value / self.maxValue
        currentNumTicks = int(currentFraction * self.maxTicks)
        if currentNumTicks > self.numTicks:
            self.numTicks = currentNumTicks
            self.fraction = currentFraction
            self.printStatus()
        self.priorValue = value

    def printStatus(self):
        sys.stdout.write("\r")
        sys.stdout.write(
            "[%-*s] %3d%%" % (self.maxTicks, self.char * self.numTicks, self.fraction * 100)
        )
        if self.numTicks == self.maxTicks:
            stopTime = time.time()
            sys.stdout.write(" (%-.1f secs elapsed)\n" % (stopTime - self.createTime))
        sys.stdout.flush()

    def finish(self):
        pass


class DataDirection(Enum):
    NONE = (0,)
    READ = (1,)
    WRITE = 2


class SpinnyThing:
    def __init__(self):
        self.chars = ["|", "/", "-", "\\"]
        self.index = 0

    def increment(self, value=1):
        sys.stdout.write("\b" + self.chars[self.index])  # pragma: no mutate
        sys.stdout.flush()
        self.index = (self.index + value) % len(self.chars)

    def finish(self):
        sys.stdout.write("\b*\n")
        sys.stdout.flush()


def iterate_progress(obj, *args, **kwargs):
    try:
        progress = ProgressBar(len(obj))
    except TypeError:
        progress = SpinnyThing()
    for o in obj:
        yield o
        progress.increment()
    progress.finish()


try:
    from tqdm import tqdm
except ImportError:
    tqdm = iterate_progress


def state(obj):
    if hasattr(obj, "state"):
        return obj.state()

    if hasattr(obj.__class__, "StateKeys"):
        rv = {}
        for key in obj.__class__.StateKeys:
            attr = key
            if isinstance(key, tuple):
                (key, attr) = key
            rv[key] = state(getattr(obj, attr))
        return rv

    if isinstance(obj, dict):
        return {k: state(v) for k, v in obj.items()}

    if isinstance(obj, (str, int, float)):
        return obj

    try:
        return [state(i) for i in obj]
    except TypeError:
        pass

    return obj


def state_key_ordering(cls):
    def tup(obj):
        return tuple([getattr(obj, k) for k in cls.StateKeys])

    def lt(a, b):
        return tup(a) < tup(b)

    def eq(a, b):
        return tup(a) == tup(b)

    cls.__lt__ = lt
    cls.__eq__ = eq

    return functools.total_ordering(cls)


def hash_combine(*objs, **kwargs):
    shift = 1
    if "shift" in kwargs:
        shift = kwargs["shift"]

    if len(objs) == 1:
        objs = objs[0]

    rv = 0
    try:
        it = iter(objs)
        rv = next(it)
        for value in it:
            rv = (rv << shift) ^ value
    except TypeError:
        return objs
    except StopIteration:
        pass
    return rv


def hash_objs(*objs, **kwargs):
    return hash(tuple(objs))


def ClientExecutionLock(lockPath: str):
    if not lockPath:
        return open(os.devnull)

    import filelock

    return filelock.FileLock(lockPath)


def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, defaultDictionary):
    if key in sourceDictionary:
        destinationDictionary[key] = deepcopy(sourceDictionary[key])
    else:
        destinationDictionary[key] = deepcopy(defaultDictionary[key])


def isRhel8() -> bool:
    """
    Check if the current OS is Red Hat Enterprise Linux 8 by reading the /etc/os-release file.

    Returns:
        True if the current OS is RHEL 8, False otherwise
    """
    file = Path("/etc/os-release")
    pattern = r'NAME="Red Hat Enterprise Linux".*VERSION_ID="8\.\d+"'
    if not file.exists():
        return False
    with open(file, "r") as f:
        content = f.read()
    match = re.search(pattern, content, re.DOTALL)
    if match:
        printWarning("Rhel8 environments may not support all tools for system queries such as amd-smi.")
        return True
    return False

########################################
# Math
########################################

def clusterEnabled(clusterDim):
    """True when a workgroup cluster is requested (ClusterDim [x, y] is not [1, 1])."""
    return (clusterDim[0] * clusterDim[1]) != 1

def isPow2(n):
    """True when ``n`` is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0

def streamKMulticast(d):
    """True when the StreamK=3 cluster multicast path is active.

    Single source of truth derived from ClusterDim: on StreamK=3 a spatial
    cluster (ClusterDim[0] = Cs > 1, i.e. Cs peers sharing B across M-adjacent
    tiles) IS the cluster multicast path, so there is no separate state key to
    store or serialize.

    StreamKForceDPOnly=1 is part of the condition, not an extra gate the callers
    add: only the DP-only schedule launches over the real M x N tile space that
    the mask derivation, the tile-index fold and the padded-peer exit assume.
    The two-tile (FDPO=0) SK3 cluster is cluster *reduction*, which predates this
    path and must keep emitting exactly what it emits without any of it.

    ``d`` may be a kernel or a solution ``state`` dict; both expose "StreamK"
    and "ClusterDim". Uses ``.get`` for partial-state derivation call sites that
    construct a dict without a StreamK / ClusterDim / StreamKForceDPOnly key.
    """
    return (d.get("StreamK", 0) == 3
            and d.get("ClusterDim", [1, 1])[0] > 1
            and bool(d.get("StreamKForceDPOnly", 0)))

def streamK2DMulticast(d):
    """True when the cluster multicasts A as well as B, i.e. Ck > 1.

    ClusterDim = [Cs, Ck] with BOTH axes > 1: Cs/X peers share B on M-adjacent
    tiles and Ck/Y peers share A on N-adjacent tiles. A 1-D [Cs, 1] cluster is
    the Ck == 1 degenerate of the same shape -- A simply has no peers there.

    ``d`` may be a kernel or a solution ``state`` dict; uses ``.get`` for
    partial-state derivation call sites.
    """
    clusterDim = d.get("ClusterDim", [1, 1])
    return clusterDim[0] > 1 and clusterDim[1] > 1

def log2(x):
    return int(log(x, 2) + 0.5)

def effectiveMatrixInstMN(matrixInstM, matrixInstN, sourceSwap):
    # Effective per-instruction M/N extents for tiling/layout. SourceSwap on a
    # non-square MatrixInstruction transposes the accumulator, so the M/N tiling
    # extents swap; the physical MatrixInstM/N (opcode / accumulator-layout source
    # of truth) are unchanged. Square MI or SS0 return the inputs unchanged.
    if sourceSwap and matrixInstM != matrixInstN:
        return matrixInstN, matrixInstM
    return matrixInstM, matrixInstN

def ceilDivide(numerator, denominator):
    # import pdb
    # pdb.set_trace()
    try:
        if numerator < 0 or denominator < 0:
            raise ValueError
    except ValueError:
        print("ERROR: Can't have a negative register value")  # pragma: no mutate
        return 0
    try:
        div = int((numerator+denominator-1) // denominator)
    except ZeroDivisionError:
        print("ERROR: Divide by 0")  # pragma: no mutate
        return 0
    return div

def roundUpToNearestMultiple(numerator, denominator):
    return ceilDivide(numerator,denominator)*int(denominator)

# Given a divisor, this routine computes the corresponding multiplicative constant
# and required post shifts.
#
# Algorithm based on: https://dl.acm.org/doi/pdf/10.1145/178243.178249
#
# Inputs:
#   d: divisor
#   N: Number of bits integers are represented in
#   p: precision in bits (usually N = P)
#
# Output:
#   mhigh: multiplicative constant
#   shPost: amount to right shift after multiplication
def choose_multiplier(d, N, p):
    l = int(math.ceil(math.log(d, 2)))
    shPost = l
    mlow = 2**(N+l) // d
    mhigh = (2**(N+l) + 2 ** (N + l - p )) // d
    while ((mlow // 2) < (mhigh // 2)) and shPost > 0:
        mlow //= 2
        mhigh //= 2
        shPost -=1
    return mhigh, shPost, l

def wmmaV3InputVgprLayout(wmma: Sequence[int], dtypeBitWidth: Optional[int] = None) -> Tuple[int]:
    # wmmaV3InputVgprLayout: (numReadsUnroll, numVecTile, numVecUnroll, NumElementPerRead)
    wmma = tuple(wmma)
    if wmma == (16, 16, 4, 1):
        return (1, 16, 2, 2)
    elif wmma == (16, 16, 32, 1):
        return (2, 16, 2, 8)
    elif wmma == (16, 16, 64, 1):
        return (2, 16, 2, 16)
    elif wmma == (16, 16, 128, 1) or wmma == (32, 16, 128, 1):
        assert dtypeBitWidth
        if dtypeBitWidth == 8:
            return (4, 16, 2, 16)
        if dtypeBitWidth == 4 or dtypeBitWidth == 6:
            return (2, 16, 2, 32)
        assert False, f"Unsupported datatype bitwidth: {dtypeBitWidth}"
    else:
        assert False, f"Unhandled WMMA: {wmma}"

# Bytes moved by one buffer_load_dwordx4, the widest global load we issue.
SWIZZLE_LOAD_BYTES = 16

def swizzleGeometry(solution, tc: str) -> dict:
    """Layout of the pre-swizzled (pre-tiled) tensor `tc` ("A" or "B").

    The tensor is a sequence of swizzle blocks; a block is one wave's global load, of
    MI_{M|N} rows each holding laneSize contiguous unroll elements.

    dupFactor is 2 where the matrix instruction replicates operands across the wave
    (gfx10/gfx11 WMMA), so only half the lanes are distinct; 1 for MFMA and gfx12.
    loadsPerLane is >1 where a lane's operand exceeds one load (gfx11 fp16: 32 bytes);
    the remainder sits in the next block along the unroll dimension.

    `solution` may be a partly derived state; only MIInputPerThread{tc}, MatrixInst{M,N,K},
    WavefrontSize and ProblemType.DataType{tc} are read.
    """
    bpe       = int(solution["ProblemType"][f"DataType{tc}"].numBytes())
    miInput   = solution[f"MIInputPerThread{tc}"]
    miMorN    = solution["MatrixInstM"] if tc == "A" else solution["MatrixInstN"]
    # Pack several MI steps into one load when one operand is narrower than a dwordx4.
    packK     = max(1, SWIZZLE_LOAD_BYTES // miInput // bpe)
    miOperand = miInput * packK
    laneSize  = min(miOperand, SWIZZLE_LOAD_BYTES // bpe)
    # Elements the wave holds vs. distinct elements the instruction consumes.
    dupFactor = max(1, (solution["WavefrontSize"] * miInput) // (miMorN * solution["MatrixInstK"]))
    lanesUsed = solution["WavefrontSize"] // dupFactor
    return {
        "packK":        packK,
        "laneSize":     laneSize,
        "swizzleK":     max(1, lanesUsed // miMorN) * laneSize,
        "lanesUsed":    lanesUsed,
        "dupFactor":    dupFactor,
        "loadsPerLane": miOperand // laneSize,
    }
