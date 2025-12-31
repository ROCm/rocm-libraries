################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from enum import Enum

class Backend(Enum):
    ROCISA = "rocisa"
    NATIVE = "native"

_CURRENT_BACKEND = Backend.ROCISA


def instruction_wrapper(rocisa_class: str, module: str = "rocisa.instruction"):
    """
    Simple decorator to create dual-backend instruction wrapper.

    Usage:
        @instruction_wrapper("DSStoreB8")
        class DSStoreB8:
            def __init__(self, dstAddr, src, ds=None, comment=""):
                self.dstAddr = dstAddr
                self.src = src
                self.ds = ds
                self.comment = comment
    """
    def decorator(cls):
        original_init = cls.__init__ if hasattr(cls, '__init__') else lambda self: None

        # Import the rocisa class once for class-level access
        mod = __import__(module, fromlist=[rocisa_class])
        rocisa_cls = getattr(mod, rocisa_class)

        # Store reference to rocisa class for class-level method access
        cls._rocisa_class = rocisa_cls

        def new_init(self, *args, **kwargs):
            self._backend = _CURRENT_BACKEND
            self._rocisa_inst = None
            self._extra_kwargs = {}

            if self._backend == Backend.ROCISA:
                # Separate kwargs that the wrapper __init__ accepts vs extras for rocisa
                import inspect
                sig = inspect.signature(original_init)
                wrapper_params = set(sig.parameters.keys()) - {'self'}

                # Split kwargs into wrapper params and extra params
                wrapper_kwargs = {}
                extra_kwargs = {}
                for k, v in kwargs.items():
                    if k in wrapper_params:
                        wrapper_kwargs[k] = v
                    else:
                        extra_kwargs[k] = v

                # Pass all kwargs to rocisa (it accepts more params than wrapper)
                self._rocisa_inst = cls._rocisa_class(*args, **kwargs)
                self._extra_kwargs = extra_kwargs

                # Only pass wrapper params to original_init
                kwargs = wrapper_kwargs

            original_init(self, *args, **kwargs)

        cls.__init__ = new_init

        if not hasattr(cls, '__str__'):
            cls.__str__ = lambda self: str(self._rocisa_inst) if self._rocisa_inst else ""

        if not hasattr(cls, '__getattr__'):
            def __getattr__(self, name):
                """Delegate attribute access to underlying rocisa instruction."""
                if self._rocisa_inst is not None and hasattr(self._rocisa_inst, name):
                    return getattr(self._rocisa_inst, name)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

            cls.__getattr__ = __getattr__

        if not hasattr(cls, '__setattr__'):
            def __setattr__(self, name, value):
                """Delegate attribute assignment to underlying rocisa instruction."""
                # Handle internal attributes directly
                if name in ('_backend', '_rocisa_inst', '_extra_kwargs'):
                    object.__setattr__(self, name, value)
                    return

                # Set on the wrapper instance first
                object.__setattr__(self, name, value)

                # Also set on the rocisa instance if it exists and has the attribute
                if hasattr(self, '_rocisa_inst') and self._rocisa_inst is not None:
                    if hasattr(self._rocisa_inst, name):
                        setattr(self._rocisa_inst, name, value)

            cls.__setattr__ = __setattr__

        # Add __getattribute__ to delegate class-level attribute access to rocisa class
        original_getattribute = cls.__getattribute__

        def new_getattribute(self_or_cls, name):
            # Try to get from the wrapper class first
            try:
                return object.__getattribute__(self_or_cls, name)
            except AttributeError:
                # If not found and we have _rocisa_class, try to get from there
                if hasattr(self_or_cls, '_rocisa_class'):
                    rocisa_cls = object.__getattribute__(self_or_cls, '_rocisa_class')
                    if hasattr(rocisa_cls, name):
                        return getattr(rocisa_cls, name)
                raise

        # Make class-level methods accessible (e.g., issueLatency())
        def __getattribute__(name):
            if name in ['__class__', '__dict__', '_rocisa_class']:
                return object.__getattribute__(cls, name)
            # Try wrapper class first
            try:
                return object.__getattribute__(cls, name)
            except AttributeError:
                # Delegate to rocisa class for class-level methods
                if hasattr(cls, '_rocisa_class') and hasattr(cls._rocisa_class, name):
                    return getattr(cls._rocisa_class, name)
                raise

        # Override __getattribute__ at metaclass level for class-level attribute access
        # This allows DSLoadB128.issueLatency() to work
        original_class_getattribute = type(cls).__getattribute__

        def class_getattribute(cls_self, name):
            try:
                return original_class_getattribute(cls_self, name)
            except AttributeError:
                if hasattr(cls_self, '_rocisa_class'):
                    rocisa_cls = object.__getattribute__(cls_self, '_rocisa_class')
                    if hasattr(rocisa_cls, name):
                        return getattr(rocisa_cls, name)
                raise

        # Create a new metaclass that overrides __getattribute__
        MetaWithDelegation = type('MetaWithDelegation', (type(cls),), {
            '__getattribute__': class_getattribute
        })

        # Recreate the class with the new metaclass
        cls = MetaWithDelegation(cls.__name__, cls.__bases__, dict(cls.__dict__))
        cls._rocisa_class = rocisa_cls
        cls.__init__ = new_init
        if not hasattr(cls, '__str__'):
            cls.__str__ = lambda self: str(self._rocisa_inst) if self._rocisa_inst else ""
        if not hasattr(cls, '__getattr__'):
            cls.__getattr__ = __getattr__

        return cls

    return decorator


def set_backend(backend: Backend):
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = backend


# ============================================================================
# Define wrappers
# ============================================================================

@instruction_wrapper("SWaitAlu")
class SWaitAlu:
    def __init__(self, va_vdst = -1, va_sdst = -1, va_ssrc = -1, hold_cnt = -1, vm_vsrc = -1, va_vcc = -1, sa_sdst = -1, comment = ""):
        self.va_vdst = va_vdst
        self.va_sdst = va_sdst
        self.va_ssrc = va_ssrc
        self.hold_cnt = hold_cnt
        self.vm_vsrc = vm_vsrc
        self.va_vcc = va_vcc
        self.sa_sdst = sa_sdst
        self.comment = comment

    def create(self, builder):
        return builder.SWaitAlu(self.va_vdst, self.va_sdst, self.va_ssrc, self.hold_cnt, self.vm_vsrc, self.va_vcc, self.sa_sdst, self.comment)

@instruction_wrapper("VMovB32")
class VMovB32:
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VMovB32(self.dst, self.src, self.comment)

@instruction_wrapper("VMovB64")
class VMovB64:
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VMovB64(self.dst, self.src, self.comment)

@instruction_wrapper("Label", "rocisa.code")
class Label:
    def __init__(self, label, comment = ""):
        self.label = label
        self.comment = comment

    def create(self, builder):
        return builder.createLabel(self.label)

@instruction_wrapper("SCmpEQU32")
class SCmpEQU32:
    """Scalar compare equal (unsigned 32-bit): SCC = (src0 == src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpEQU32(self.src0, self.src1, self.comment)

@instruction_wrapper("SCmpLeU32")
class SCmpLeU32:
    """Scalar compare less or equal (unsigned 32-bit): SCC = (src0 <= src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpLeU32(self.src0, self.src1, self.comment)

@instruction_wrapper("SCmpGeU32")
class SCmpGeU32:
    """Scalar compare greater or equal (unsigned 32-bit): SCC = (src0 >= src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpGeU32(self.src0, self.src1, self.comment)

# ============================================================================
# SALU (Scalar ALU) Instructions
# ============================================================================

# Scalar Comparison Instructions
@instruction_wrapper("SCmpLtU32")
class SCmpLtU32:
    """Scalar compare: SCC = (src0 < src1) unsigned"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpLtU32(self.src0, self.src1, self.comment)

@instruction_wrapper("SCmpGtU32")
class SCmpGtU32:
    """Scalar compare: SCC = (src0 > src1) unsigned"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpGtU32(self.src0, self.src1, self.comment)

@instruction_wrapper("SCmpLgU32")
class SCmpLgU32:
    """Scalar compare: SCC = (src0 != src1) unsigned"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpLgU32(self.src0, self.src1, self.comment)

@instruction_wrapper("SCmpEQI32")
class SCmpEQI32:
    """Scalar compare: SCC = (src0 == src1) signed"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpEQI32(self.src0, self.src1, self.comment)

@instruction_wrapper("SCmpLtI32")
class SCmpLtI32:
    """Scalar compare: SCC = (src0 < src1) signed"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpLtI32(self.src0, self.src1, self.comment)

@instruction_wrapper("SCmpGtI32")
class SCmpGtI32:
    """Scalar compare: SCC = (src0 > src1) signed"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpGtI32(self.src0, self.src1, self.comment)

# Scalar Arithmetic Instructions
@instruction_wrapper("SMulI32")
class SMulI32:
    """Scalar 32-bit signed multiply: dst = src0 * src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SMulI32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SCBranchSCC1")
class SCBranchSCC1:
    """Conditional branch if SCC == 1"""
    def __init__(self, labelName, comment = ""):
        self.labelName = labelName
        self.comment = comment

    def create(self, builder):
        return builder.SCBranchSCC1(self.labelName, self.comment)

@instruction_wrapper("SCBranchSCC0")
class SCBranchSCC0:
    """Conditional branch if SCC == 0"""
    def __init__(self, labelName, comment = ""):
        self.labelName = labelName
        self.comment = comment

    def create(self, builder):
        return builder.SCBranchSCC0(self.labelName, self.comment)

@instruction_wrapper("SSetRegIMM32B32")
class SSetRegIMM32B32:
    """Set hardware register to immediate 32-bit value"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SSetRegIMM32B32(self.dst, self.src, self.comment)

# Scalar Move Instructions
@instruction_wrapper("SMovB32")
class SMovB32:
    """Scalar 32-bit move"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SMovB32(self.dst, self.src, self.comment)

@instruction_wrapper("SMovB64")
class SMovB64:
    """Scalar 64-bit move"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SMovB64(self.dst, self.src, self.comment)

# Scalar Arithmetic Instructions
@instruction_wrapper("SAddU32")
class SAddU32:
    """Scalar 32-bit unsigned add"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SAddU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SAddI32")
class SAddI32:
    """Scalar 32-bit signed add"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SAddI32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SAddCU32")
class SAddCU32:
    """Scalar 32-bit add with carry"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SAddCU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SSubU32")
class SSubU32:
    """Scalar 32-bit unsigned subtract"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SSubU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SSubBU32")
class SSubBU32:
    """Scalar 32-bit subtract with borrow"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SSubBU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SMulLOU32")
class SMulLOU32:
    """Scalar 32-bit multiply low (unsigned)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SMulLOU32(self.dst, self.src0, self.src1, self.comment)

# Scalar Logical Instructions
@instruction_wrapper("SAndB32")
class SAndB32:
    """Scalar 32-bit bitwise AND"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SAndB32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SAndB64")
class SAndB64:
    """Scalar 64-bit bitwise AND"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SAndB64(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SOrB32")
class SOrB32:
    """Scalar 32-bit bitwise OR"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SOrB32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SOrB64")
class SOrB64:
    """Scalar 64-bit bitwise OR"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SOrB64(self.dst, self.src0, self.src1, self.comment)

# Scalar Shift Instructions
@instruction_wrapper("SLShiftLeftB32")
class SLShiftLeftB32:
    """Scalar 32-bit logical shift left"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SLShiftLeftB32(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("SLShiftLeftB64")
class SLShiftLeftB64:
    """Scalar 64-bit logical shift left"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        # rocisa expects (dst, shiftHex, src, comment)
        return builder.SLShiftLeftB64(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("SLShiftRightB32")
class SLShiftRightB32:
    """Scalar 32-bit logical shift right"""
    def __init__(self, dst, shiftHex, src , comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SLShiftRightB32(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("SLShiftRightB64")
class SLShiftRightB64:
    """Scalar 64-bit logical shift right"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SLShiftRightB64(self.dst, self.shiftHex, self.src, self.comment)

# Scalar Conditional Move Instructions
@instruction_wrapper("SCMovB32")
class SCMovB32:
    """Scalar 32-bit conditional move (if SCC == 1)"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SCMovB32(self.dst, self.src, self.comment)

@instruction_wrapper("SCSelectB32")
class SCSelectB32:
    """Scalar 32-bit select (dst = SCC ? src0 : src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCSelectB32(self.dst, self.src0, self.src1, self.comment)

# Scalar Control Flow Instructions
@instruction_wrapper("SBarrier")
class SBarrier:
    """Scalar barrier - synchronize all waves in a workgroup"""
    def __init__(self, comment = ""):
        self.comment = comment

    def create(self, builder):
        return builder.SBarrier(self.comment)

@instruction_wrapper("SBranch")
class SBranch:
    """Unconditional branch"""
    def __init__(self, labelName, comment = ""):
        self.labelName = labelName
        self.comment = comment

    def create(self, builder):
        return builder.SBranch(self.labelName, self.comment)

@instruction_wrapper("SNop")
class SNop:
    """Scalar no operation (with optional wait count)"""
    def __init__(self, waitState, comment = ""):
        self.waitState = waitState
        self.comment = comment

    def create(self, builder):
        return builder.SNop(self.waitState, self.comment)

@instruction_wrapper("SWaitCnt")
class SWaitCnt:
    """Scalar wait for outstanding memory operations"""
    def __init__(self, vlcnt=-1, vscnt=-1, dscnt=-1, kmcnt=-1, comment="", waitAll=False):
        self.vlcnt = vlcnt
        self.vscnt = vscnt
        self.dscnt = dscnt
        self.kmcnt = kmcnt
        self.comment = comment
        self.waitAll = waitAll

    def create(self, builder):
        return builder.SWaitCnt(vlcnt=self.vlcnt, vscnt=self.vscnt, dscnt=self.dscnt, kmcnt=self.kmcnt, comment=self.comment, waitAll=self.waitAll)

@instruction_wrapper("SWaitTensorcnt")
class SWaitTensorcnt:
    """Scalar wait for tensor memory operations"""
    def __init__(self, tensorcnt=-1, comment=""):
        self.tensorcnt = tensorcnt
        self.comment = comment

    def create(self, builder):
        return builder.SWaitTensorcnt(tensorcnt=self.tensorcnt, comment=self.comment)

# Scalar Execution Mask Instructions
@instruction_wrapper("SOrSaveExecB32")
class SOrSaveExecB32:
    """Save and modify 32-bit exec mask: dst = exec, exec = exec | src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SOrSaveExecB32(self.dst, self.src, self.comment)

@instruction_wrapper("SOrSaveExecB64")
class SOrSaveExecB64:
    """Save and modify 64-bit exec mask: dst = exec, exec = exec | src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SOrSaveExecB64(self.dst, self.src, self.comment)

@instruction_wrapper("SSetMask")
class SSetMask:
    """Set execution mask"""
    def __init__(self, src, comment = ""):
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SSetMask(self.src, self.comment)

@instruction_wrapper("SSetPrior")
class SSetPrior:
    """Set wave priority"""
    def __init__(self, prior, comment = ""):
        self.prior = prior
        self.comment = comment

    def create(self, builder):
        return builder.SSetPrior(self.prior, self.comment)

# Scalar Memory Instructions
@instruction_wrapper("SLoadB32")
class SLoadB32:
    """Scalar 32-bit memory load"""
    def __init__(self, dst, src, offset = 0, comment = ""):
        self.dst = dst
        self.src = src
        self.offset = offset
        self.comment = comment

    def create(self, builder):
        return builder.SLoadB32(self.dst, self.src, self.offset, self.comment)

# Special Scalar Instructions
@instruction_wrapper("SMulInt64to32")
class SMulInt64to32:
    """Scalar multiply returning 32-bit result from 64-bit operation"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SMulInt64to32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VReadfirstlaneB32")
class VReadfirstlaneB32:
    """Read from first active lane in VGPR to SGPR"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VReadfirstlaneB32(self.dst, self.src, self.comment)

# ============================================================================
# VALU (Vector ALU) Instructions
# ============================================================================

# V_ADD Instructions
@instruction_wrapper("VAdd3U32")
class VAdd3U32:
    """3-way add: dst = src0 + src1 + src2"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VAdd3U32(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VAddU32")
class VAddU32:
    """32-bit unsigned add: dst = src0 + src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VAddI32")
class VAddI32:
    """32-bit signed add: dst = src0 + src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddI32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VAddF16")
class VAddF16:
    """FP16 add: dst = src0 + src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VAddF32")
class VAddF32:
    """FP32 add: dst = src0 + src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddF32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VAddF64")
class VAddF64:
    """FP64 add: dst = src0 + src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddF64(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VAddCOU32")
class VAddCOU32:
    """32-bit add with carry out: dst = src0 + src1, dst1 = carry"""
    def __init__(self, dst, dst1, src0, src1, comment = ""):
        self.dst = dst
        self.dst1 = dst1  # Carry-out destination
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddCOU32(self.dst, self.dst1, self.src0, self.src1, self.comment)

@instruction_wrapper("VAddLShiftLeftU32")
class VAddLShiftLeftU32:
    """Add with left shift: dst = src0 + (src1 << shift)"""
    def __init__(self, dst, shiftHex, src0, src1, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddLShiftLeftU32(self.dst, self.shiftHex, self.src0, self.src1, self.comment)

@instruction_wrapper("VAddPKF16")
class VAddPKF16:
    """Packed FP16 add"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddPKF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VAddPKF32")
class VAddPKF32:
    """Packed FP32 add"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAddPKF32(self.dst, self.src0, self.src1, self.comment)

# V_SUB Instructions
@instruction_wrapper("VSubU32")
class VSubU32:
    """32-bit unsigned subtract: dst = src0 - src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VSubU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VSubI32")
class VSubI32:
    """32-bit signed subtract: dst = src0 - src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VSubI32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VSubF32")
class VSubF32:
    """FP32 subtract: dst = src0 - src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VSubF32(self.dst, self.src0, self.src1, self.comment)

# V_MUL Instructions
@instruction_wrapper("VMulF16")
class VMulF16:
    """FP16 multiply: dst = src0 * src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMulF32")
class VMulF32:
    """FP32 multiply: dst = src0 * src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulF32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMulF64")
class VMulF64:
    """FP64 multiply: dst = src0 * src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulF64(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMulLOU32")
class VMulLOU32:
    """32-bit multiply low: dst = src0 * src1 (lower 32 bits)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulLOU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMulHIU32")
class VMulHIU32:
    """32-bit multiply high: dst = (src0 * src1) >> 32"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulHIU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMulI32I24")
class VMulI32I24:
    """24-bit signed multiply: dst = src0[23:0] * src1[23:0]"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulI32I24(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMulU32U24")
class VMulU32U24:
    """24-bit unsigned multiply: dst = src0[23:0] * src1[23:0]"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulU32U24(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMulPKF16")
class VMulPKF16:
    """Packed FP16 multiply"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulPKF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMulPKF32")
class VMulPKF32:
    """Packed FP32 multiply"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMulPKF32(self.dst, self.src0, self.src1, self.comment)

# V_FMA/MAC Instructions
@instruction_wrapper("VFmaF16")
class VFmaF16:
    """FP16 fused multiply-add: dst = src0 * src1 + src2"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VFmaF16(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VFmaF32")
class VFmaF32:
    """FP32 fused multiply-add: dst = src0 * src1 + src2"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VFmaF32(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VFmaMixF32")
class VFmaMixF32:
    """Mixed precision FMA"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VFmaMixF32(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VFmaPKF16")
class VFmaPKF16:
    """Packed FP16 FMA"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VFmaPKF16(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VMadI32I24")
class VMadI32I24:
    """24-bit signed MAD: dst = src0[23:0] * src1[23:0] + src2"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VMadI32I24(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VMadU32U24")
class VMadU32U24:
    """24-bit unsigned MAD: dst = src0[23:0] * src1[23:0] + src2"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VMadU32U24(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VMadMixF32")
class VMadMixF32:
    """Mixed precision MAD"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VMadMixF32(self.dst, self.src0, self.src1, self.src2, self.comment)

# Dot Product Instructions


# Logical Instructions
@instruction_wrapper("VAndB32")
class VAndB32:
    """Bitwise AND: dst = src0 & src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VAndB32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VOrB32")
class VOrB32:
    """Bitwise OR: dst = src0 | src1"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VOrB32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VAndOrB32")
class VAndOrB32:
    """Bitwise AND-OR: dst = (src0 & src1) | src2"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VAndOrB32(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VNotB32")
class VNotB32:
    """Bitwise NOT: dst = ~src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VNotB32(self.dst, self.src, self.comment)

# Shift Instructions
@instruction_wrapper("VLShiftLeftB16")
class VLShiftLeftB16:
    """16-bit left shift: dst = src0 << src1"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VLShiftLeftB16(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("VLShiftLeftB32")
class VLShiftLeftB32:
    """32-bit left shift: dst = src0 << src1"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VLShiftLeftB32(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("VLShiftLeftB64")
class VLShiftLeftB64:
    """64-bit left shift: dst = src0 << src1"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VLShiftLeftB64(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("VLShiftRightB32")
class VLShiftRightB32:
    """32-bit logical right shift: dst = src0 >> src1"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VLShiftRightB32(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("VLShiftRightB64")
class VLShiftRightB64:
    """64-bit logical right shift: dst = src0 >> src1"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VLShiftRightB64(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("VAShiftRightI32")
class VAShiftRightI32:
    """32-bit arithmetic right shift: dst = src0 >> src1 (sign-extended)"""
    def __init__(self, dst, shiftHex, src, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VAShiftRightI32(self.dst, self.shiftHex, self.src, self.comment)

@instruction_wrapper("VLShiftLeftAddU32")
class VLShiftLeftAddU32:
    """Left shift and add: dst = src0 + (src1 << shift)"""
    def __init__(self, dst, shiftHex, src0, src1, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VLShiftLeftAddU32(self.dst, self.shiftHex, self.src0, self.src1, self.comment)

@instruction_wrapper("VLShiftLeftOrB32")
class VLShiftLeftOrB32:
    """Left shift and OR: dst = src0 | (src1 << src2)"""
    def __init__(self, dst, shiftHex, src0, src1, comment = ""):
        self.dst = dst
        self.shiftHex = shiftHex
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VLShiftLeftOrB32(self.dst, self.shiftHex, self.src0, self.src1, self.comment)

# Bit Field Instructions
@instruction_wrapper("VBfeU32")
class VBfeU32:
    """Bit field extract unsigned: dst = (src0 >> src1) & ((1 << src2) - 1)"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VBfeU32(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VBfeI32")
class VBfeI32:
    """Bit field extract signed"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VBfeI32(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VBfiB32")
class VBfiB32:
    """Bit field insert"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VBfiB32(self.dst, self.src0, self.src1, self.src2, self.comment)

# Comparison Instructions
@instruction_wrapper("VCmpEQU32")
class VCmpEQU32:
    """Compare equal unsigned: VCC = (src0 == src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpEQU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpEQI32")
class VCmpEQI32:
    """Compare equal signed: VCC = (src0 == src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpEQI32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpNeU32")
class VCmpNeU32:
    """Compare not equal unsigned: VCC = (src0 != src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpNeU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpNeU64")
class VCmpNeU64:
    """Compare not equal unsigned 64-bit: VCC = (src0 != src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpNeU64(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpLtU32")
class VCmpLtU32:
    """Compare less than unsigned: dst = (src0 < src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpLtU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpLtI32")
class VCmpLtI32:
    """Compare less than signed: VCC = (src0 < src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpLtI32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpLeI32")
class VCmpLeI32:
    """Compare less or equal signed: VCC = (src0 <= src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpLeI32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGTI32")
class VCmpGTI32:
    """Compare greater than signed: VCC = (src0 > src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGTI32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGtU32")
class VCmpGtU32:
    """Compare greater than unsigned: VCC = (src0 > src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGtU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGEU32")
class VCmpGEU32:
    """Compare greater or equal unsigned: VCC = (src0 >= src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGEI32")
class VCmpGEI32:
    """Compare greater or equal signed: VCC = (src0 >= src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEI32(self.src0, self.src1, self.comment)

# FP Comparisons
@instruction_wrapper("VCmpGEF16")
class VCmpGEF16:
    """FP16 compare greater or equal: VCC = (src0 >= src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEF16(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGEF32")
class VCmpGEF32:
    """FP32 compare greater or equal: VCC = (src0 >= src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGEF64")
class VCmpGEF64:
    """FP64 compare greater or equal: VCC = (src0 >= src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEF64(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGTF16")
class VCmpGTF16:
    """FP16 compare greater than: VCC = (src0 > src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGTF16(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGTF32")
class VCmpGTF32:
    """FP32 compare greater than: VCC = (src0 > src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGTF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGTF64")
class VCmpGTF64:
    """FP64 compare greater than: VCC = (src0 > src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGTF64(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpUF32")
class VCmpUF32:
    """FP32 compare unordered: VCC = isnan(src0) || isnan(src1)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpUF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpClassF32")
class VCmpClassF32:
    """FP32 class test"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpClassF32(self.src0, self.src1, self.comment)

# Compare with Exec Mask Update (CMPX)
@instruction_wrapper("VCmpXEqU32")
class VCmpXEqU32:
    """Compare equal with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXEqU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXNeU16")
class VCmpXNeU16:
    """Compare not equal 16-bit with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXNeU16(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXNeU32")
class VCmpXNeU32:
    """Compare not equal with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXNeU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXLtU32")
class VCmpXLtU32:
    """Compare less than unsigned with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXLtU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXLtU64")
class VCmpXLtU64:
    """Compare less than unsigned 64-bit with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXLtU64(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXLtI32")
class VCmpXLtI32:
    """Compare less than signed with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXLtI32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXLtF32")
class VCmpXLtF32:
    """FP32 compare less than with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXLtF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXLeU32")
class VCmpXLeU32:
    """Compare less or equal unsigned with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXLeU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXGeU32")
class VCmpXGeU32:
    """Compare greater or equal unsigned with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXGeU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXGtU32")
class VCmpXGtU32:
    """Compare greater than unsigned with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXGtU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXClassF32")
class VCmpXClassF32:
    """FP32 class test with exec mask update"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXClassF32(self.src0, self.src1, self.comment)

# Conditional Move
@instruction_wrapper("VCndMaskB32")
class VCndMaskB32:
    """Conditional move: dst = VCC ? src1 : src0"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCndMaskB32(self.dst, self.src0, self.src1, self.comment)

# Min/Max Instructions
@instruction_wrapper("VMinI32")
class VMinI32:
    """Signed minimum: dst = min(src0, src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMinI32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMaxI32")
class VMaxI32:
    """Signed maximum: dst = max(src0, src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMaxI32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMinF16")
class VMinF16:
    """FP16 minimum: dst = min(src0, src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMinF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMinF32")
class VMinF32:
    """FP32 minimum: dst = min(src0, src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMinF32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMinF64")
class VMinF64:
    """FP64 minimum: dst = min(src0, src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMinF64(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMaxF16")
class VMaxF16:
    """FP16 maximum: dst = max(src0, src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMaxF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMaxF32")
class VMaxF32:
    """FP32 maximum: dst = max(src0, src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMaxF32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMaxF64")
class VMaxF64:
    """FP64 maximum: dst = max(src0, src1)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMaxF64(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMaxPKF16")
class VMaxPKF16:
    """Packed FP16 maximum"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VMaxPKF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VMed3I32")
class VMed3I32:
    """Signed median of 3: dst = median(src0, src1, src2)"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VMed3I32(self.dst, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("VMed3F32")
class VMed3F32:
    """FP32 median of 3: dst = median(src0, src1, src2)"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VMed3F32(self.dst, self.src0, self.src1, self.src2, self.comment)

# Type Conversion Instructions
@instruction_wrapper("VCvtF32toF16")
class VCvtF32toF16:
    """Convert FP32 to FP16: dst = (fp16)src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtF32toF16(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtF16toF32")
class VCvtF16toF32:
    """Convert FP16 to FP32: dst = (fp32)src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtF16toF32(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtBF16toFP32")
class VCvtBF16toFP32:
    """Convert BF16 to FP32: dst = (fp32)src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtBF16toFP32(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtF32toI32")
class VCvtF32toI32:
    """Convert FP32 to signed int32: dst = (int32)src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtF32toI32(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtI32toF32")
class VCvtI32toF32:
    """Convert signed int32 to FP32: dst = (fp32)src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtI32toF32(self.dst, self.src, self.comment)

# Packed Conversion Instructions
@instruction_wrapper("VCvtPkF32toF16")
class VCvtPkF32toF16:
    """Convert packed FP32 to FP16"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCvtPkF32toF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VCvtPkF32toBF16")
class VCvtPkF32toBF16:
    """Convert packed FP32 to BF16"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCvtPkF32toBF16(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VCvtPkF32toFP8")
class VCvtPkF32toFP8:
    """Convert packed FP32 to FP8"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCvtPkF32toFP8(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VCvtPkF32toBF8")
class VCvtPkF32toBF8:
    """Convert packed FP32 to BF8"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCvtPkF32toBF8(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("VCvtPkFP8toF16")
class VCvtPkFP8toF16:
    """Convert packed FP8 to FP16"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtPkFP8toF16(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtPkFP8toF32")
class VCvtPkFP8toF32:
    """Convert packed FP8 to FP32"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtPkFP8toF32(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtPkBF8toF32")
class VCvtPkBF8toF32:
    """Convert packed BF8 to FP32"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtPkBF8toF32(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtFP8toF16")
class VCvtFP8toF16:
    """Convert FP8 to FP16"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtFP8toF16(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtFP8toF32")
class VCvtFP8toF32:
    """Convert FP8 to FP32"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtFP8toF32(self.dst, self.src, self.comment)

@instruction_wrapper("VCvtBF8toF32")
class VCvtBF8toF32:
    """Convert BF8 to FP32"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VCvtBF8toF32(self.dst, self.src, self.comment)

# Scaled Conversion Instructions
@instruction_wrapper("VCvtSRF32toFP8")
class VCvtSRF32toFP8:
    """Convert FP32 to FP8 with scaling and rounding"""
    def __init__(self, dst, src, scale, comment = ""):
        self.dst = dst
        self.src = src
        self.scale = scale
        self.comment = comment

    def create(self, builder):
        return builder.VCvtSRF32toFP8(self.dst, self.src, self.scale, self.comment)

@instruction_wrapper("VCvtSRF32toBF8")
class VCvtSRF32toBF8:
    """Convert FP32 to BF8 with scaling and rounding"""
    def __init__(self, dst, src, scale, comment = ""):
        self.dst = dst
        self.src = src
        self.scale = scale
        self.comment = comment

    def create(self, builder):
        return builder.VCvtSRF32toBF8(self.dst, self.src, self.scale, self.comment)

@instruction_wrapper("VCvtScaleFP8toF16")
class VCvtScaleFP8toF16:
    """Convert FP8 to FP16 with scaling"""
    def __init__(self, dst, src, scale, comment = ""):
        self.dst = dst
        self.src = src
        self.scale = scale
        self.comment = comment

    def create(self, builder):
        return builder.VCvtScaleFP8toF16(self.dst, self.src, self.scale, self.comment)

@instruction_wrapper("VCvtScalePkFP8toF16")
class VCvtScalePkFP8toF16:
    """Convert packed FP8 to FP16 with scaling"""
    def __init__(self, dst, src, scale, comment = ""):
        self.dst = dst
        self.src = src
        self.scale = scale
        self.comment = comment

    def create(self, builder):
        return builder.VCvtScalePkFP8toF16(self.dst, self.src, self.scale, self.comment)

@instruction_wrapper("VCvtScalePkF16toFP8")
class VCvtScalePkF16toFP8:
    """Convert packed FP16 to FP8 with scaling"""
    def __init__(self, dst, src, scale, comment = ""):
        self.dst = dst
        self.src = src
        self.scale = scale
        self.comment = comment

    def create(self, builder):
        return builder.VCvtScalePkF16toFP8(self.dst, self.src, self.scale, self.comment)

@instruction_wrapper("VCvtScalePkF16toBF8")
class VCvtScalePkF16toBF8:
    """Convert packed FP16 to BF8 with scaling"""
    def __init__(self, dst, src, scale, comment = ""):
        self.dst = dst
        self.src = src
        self.scale = scale
        self.comment = comment

    def create(self, builder):
        return builder.VCvtScalePkF16toBF8(self.dst, self.src, self.scale, self.comment)

# Pack Instructions
@instruction_wrapper("VPackF16toB32")
class VPackF16toB32:
    """Pack two FP16 values into one 32-bit register"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VPackF16toB32(self.dst, self.src0, self.src1, self.comment)

# Permute/Shuffle Instructions
@instruction_wrapper("VPermB32")
class VPermB32:
    """Byte permute"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VPermB32(self.dst, self.src0, self.src1, self.src2, self.comment)

# Math Functions
@instruction_wrapper("VExpF16")
class VExpF16:
    """FP16 exponential: dst = exp(src)"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VExpF16(self.dst, self.src, self.comment)

@instruction_wrapper("VExpF32")
class VExpF32:
    """FP32 exponential: dst = exp(src)"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VExpF32(self.dst, self.src, self.comment)

@instruction_wrapper("VRcpF16")
class VRcpF16:
    """FP16 reciprocal: dst = 1.0 / src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VRcpF16(self.dst, self.src, self.comment)

@instruction_wrapper("VRcpF32")
class VRcpF32:
    """FP32 reciprocal: dst = 1.0 / src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VRcpF32(self.dst, self.src, self.comment)

@instruction_wrapper("VRndneF32")
class VRndneF32:
    """FP32 round to nearest even: dst = round_even(src)"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VRndneF32(self.dst, self.src, self.comment)

# Special Instructions
@instruction_wrapper("VReadfirstlaneB32")
class VReadfirstlaneB32:
    """Read first active lane of VGPR to SGPR"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VReadfirstlaneB32(self.dst, self.src, self.comment)

@instruction_wrapper("VPrngB32")
class VPrngB32:
    """Pseudo-random number generator"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VPrngB32(self.dst, self.src, self.comment)

@instruction_wrapper("VNop")
class VNop:
    """Vector no-operation"""
    def __init__(self, waitCount = 0, comment = ""):
        self.waitCount = waitCount
        self.comment = comment

    def create(self, builder):
        return builder.VNop(self.waitCount, self.comment)

# ACCVGPR Instructions
@instruction_wrapper("VAccvgprReadB32")
class VAccvgprReadB32:
    """Read from accumulation VGPR"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VAccvgprReadB32(self.dst, self.src, self.comment)

@instruction_wrapper("VAccvgprWriteB32")
class VAccvgprWriteB32:
    """Write to accumulation VGPR"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VAccvgprWriteB32(self.dst, self.src, self.comment)

# Additional VALU Comparison Instructions
@instruction_wrapper("VCmpEQU32")
class VCmpEQU32:
    """Vector compare equal (unsigned 32-bit)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpEQU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXEqU32")
class VCmpXEqU32:
    """Vector compare equal with exec mask (unsigned 32-bit)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXEqU32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGEI32")
class VCmpGEI32:
    """Vector compare greater or equal (signed 32-bit)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEI32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGTI32")
class VCmpGTI32:
    """Vector compare greater than (signed 32-bit)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGTI32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGEF16")
class VCmpGEF16:
    """Vector compare greater or equal (FP16)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEF16(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGEF32")
class VCmpGEF32:
    """Vector compare greater or equal (FP32)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGEF64")
class VCmpGEF64:
    """Vector compare greater or equal (FP64)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGEF64(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGTF16")
class VCmpGTF16:
    """Vector compare greater than (FP16)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGTF16(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGTF32")
class VCmpGTF32:
    """Vector compare greater than (FP32)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGTF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpGTF64")
class VCmpGTF64:
    """Vector compare greater than (FP64)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpGTF64(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXClassF32")
class VCmpXClassF32:
    """Vector compare class (FP32) with exec mask"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXClassF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpXLtF32")
class VCmpXLtF32:
    """Vector compare less than (FP32) with exec mask"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpXLtF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpClassF32")
class VCmpClassF32:
    """Vector compare class (FP32)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpClassF32(self.src0, self.src1, self.comment)

@instruction_wrapper("VCmpUF32")
class VCmpUF32:
    """Vector compare unordered (FP32)"""
    def __init__(self, src0, src1, comment = ""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VCmpUF32(self.src0, self.src1, self.comment)

# Conditional Instructions
@instruction_wrapper("VCndMaskB32")
class VCndMaskB32:
    """Vector conditional mask: dst = src2 ? src0 : src1"""
    def __init__(self, dst, src0, src1, src2 = None, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        if self.src2 is None:
            return builder.VCndMaskB32(self.dst, self.src0, self.src1, comment=self.comment)
        return builder.VCndMaskB32(self.dst, self.src0, self.src1, self.src2, comment=self.comment)

# Carry Operations
# Logical Operations
@instruction_wrapper("VOrB32")
class VOrB32:
    """Vector bitwise OR (32-bit)"""
    def __init__(self, dst, src0, src1, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.VOrB32(self.dst, self.src0, self.src1, self.comment)

# Permute/Shuffle
@instruction_wrapper("VPermB32")
class VPermB32:
    """Vector byte permute"""
    def __init__(self, dst, src0, src1, src2, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VPermB32(self.dst, self.src0, self.src1, self.src2, self.comment)

# Dot Product Instructions
@instruction_wrapper("VDot2F32F16")
class VDot2F32F16:
    """Dot product of 2 FP16 values to FP32"""
    def __init__(self, dst, src0, src1, src2, vop3 = None, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.vop3 = vop3
        self.comment = comment

    def create(self, builder):
        if self.vop3 is None:
            return builder.VDot2F32F16(self.dst, self.src0, self.src1, self.src2, comment=self.comment)
        return builder.VDot2F32F16(self.dst, self.src0, self.src1, self.src2, self.vop3, self.comment)

@instruction_wrapper("VDot2CF32F16")
class VDot2CF32F16:
    """Complex dot product of 2 FP16 values to FP32"""
    def __init__(self, dst, src0, src1, sdwa = None, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.sdwa = sdwa
        self.comment = comment

    def create(self, builder):
        if self.sdwa is None:
            return builder.VDot2CF32F16(self.dst, self.src0, self.src1, comment=self.comment)
        return builder.VDot2CF32F16(self.dst, self.src0, self.src1, self.sdwa, self.comment)

@instruction_wrapper("VDot2F32BF16")
class VDot2F32BF16:
    """Dot product of 2 BF16 values to FP32"""
    def __init__(self, dst, src0, src1, src2, vop3 = None, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.vop3 = vop3
        self.comment = comment

    def create(self, builder):
        if self.vop3 is None:
            return builder.VDot2F32BF16(self.dst, self.src0, self.src1, self.src2, comment=self.comment)
        return builder.VDot2F32BF16(self.dst, self.src0, self.src1, self.src2, self.vop3, self.comment)

@instruction_wrapper("VDot2CF32BF16")
class VDot2CF32BF16:
    """Complex dot product of 2 BF16 values to FP32"""
    def __init__(self, dst, src0, src1, sdwa = None, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.sdwa = sdwa
        self.comment = comment

    def create(self, builder):
        if self.sdwa is None:
            return builder.VDot2CF32BF16(self.dst, self.src0, self.src1, comment=self.comment)
        return builder.VDot2CF32BF16(self.dst, self.src0, self.src1, self.sdwa, self.comment)

# MAC and FMA Instructions
@instruction_wrapper("VMacF32")
class VMacF32:
    """Vector multiply-accumulate (FP32): dst = src0 * src1 + dst"""
    def __init__(self, dst, src0, src1, vop3 = None, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.vop3 = vop3
        self.comment = comment

    def create(self, builder):
        if self.vop3 is None:
            return builder.VMacF32(self.dst, self.src0, self.src1, comment=self.comment)
        return builder.VMacF32(self.dst, self.src0, self.src1, self.vop3, self.comment)

@instruction_wrapper("VFmaF64")
class VFmaF64:
    """Vector fused multiply-add (FP64): dst = src0 * src1 + src2"""
    def __init__(self, dst, src0, src1, src2, vop3 = None, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.vop3 = vop3
        self.comment = comment

    def create(self, builder):
        if self.vop3 is None:
            return builder.VFmaF64(self.dst, self.src0, self.src1, self.src2, comment=self.comment)
        return builder.VFmaF64(self.dst, self.src0, self.src1, self.src2, self.vop3, self.comment)

# Exponential Instructions
@instruction_wrapper("VExpF16")
class VExpF16:
    """Vector exponential (FP16): dst = 2^src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VExpF16(self.dst, self.src, self.comment)

@instruction_wrapper("VExpF32")
class VExpF32:
    """Vector exponential (FP32): dst = 2^src"""
    def __init__(self, dst, src, comment = ""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.VExpF32(self.dst, self.src, self.comment)

# ============================================================================
# DS (LDS - Local Data Share) Instructions
# ============================================================================

# DS Load Instructions
@instruction_wrapper("DSLoadB32")
class DSLoadB32:
    """LDS load 32-bit"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoadB32(self.dst, self.src, comment=self.comment)
        return builder.DSLoadB32(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadB64")
class DSLoadB64:
    """LDS load 64-bit"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoadB64(self.dst, self.src, comment=self.comment)
        return builder.DSLoadB64(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadB128")
class DSLoadB128:
    """LDS load 128-bit"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoadB128(self.dst, self.src, comment=self.comment)
        return builder.DSLoadB128(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadB192")
class DSLoadB192:
    """LDS load 192-bit"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        return None

@instruction_wrapper("DSLoadB64TrB16")
class DSLoadB64TrB16:
    """LDS load 64-bit transpose B16"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoadB64TrB16(self.dst, self.src, comment=self.comment)
        return builder.DSLoadB64TrB16(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadB128TrB16")
class DSLoadB128TrB16:
    """LDS load 128-bit transpose B16"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoadB128TrB16(self.dst, self.src, comment=self.comment)
        return builder.DSLoadB128TrB16(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadB64TrB8")
class DSLoadB64TrB8:
    """LDS load 64-bit transpose B8"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoadB64TrB8(self.dst, self.src, comment=self.comment)
        return builder.DSLoadB64TrB8(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadB64TrB4")
class DSLoadB64TrB4:
    """LDS load 64-bit transpose B4"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoadB64TrB4(self.dst, self.src, comment=self.comment)
        return builder.DSLoadB64TrB4(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadB96TrB6")
class DSLoadB96TrB6:
    """LDS load 96-bit transpose B6"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoadB96TrB6(self.dst, self.src, comment=self.comment)
        return builder.DSLoadB96TrB6(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoad2B32")
class DSLoad2B32:
    """LDS load 2x32-bit"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoad2B32(self.dst, self.src, comment=self.comment)
        return builder.DSLoad2B32(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoad2B64")
class DSLoad2B64:
    """LDS load 2x64-bit"""
    def __init__(self, dst, src, ds = None, comment = ""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSLoad2B64(self.dst, self.src, comment=self.comment)
        return builder.DSLoad2B64(self.dst, self.src, self.ds, self.comment)

# DS Store Instructions
@instruction_wrapper("DSStoreB8")
class DSStoreB8:
    """LDS store 8-bit"""
    def __init__(self, dstAddr, src, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStoreB8(self.dstAddr, self.src, comment=self.comment)
        return builder.DSStoreB8(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("DSStoreB16")
class DSStoreB16:
    """LDS store 16-bit"""
    def __init__(self, dstAddr, src, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStoreB16(self.dstAddr, self.src, comment=self.comment)
        return builder.DSStoreB16(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("DSStoreB32")
class DSStoreB32:
    """LDS store 32-bit"""
    def __init__(self, dstAddr, src, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStoreB32(self.dstAddr, self.src, comment=self.comment)
        return builder.DSStoreB32(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("DSStoreB64")
class DSStoreB64:
    """LDS store 64-bit"""
    def __init__(self, dstAddr, src, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStoreB64(self.dstAddr, self.src, comment=self.comment)
        return builder.DSStoreB64(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("DSStoreB96")
class DSStoreB96:
    """LDS store 96-bit"""
    def __init__(self, dstAddr, src, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStoreB96(self.dstAddr, self.src, comment=self.comment)
        return builder.DSStoreB96(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("DSStoreB128")
class DSStoreB128:
    """LDS store 128-bit"""
    def __init__(self, dstAddr, src, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStoreB128(self.dstAddr, self.src, comment=self.comment)
        return builder.DSStoreB128(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("DSStoreB192")
class DSStoreB192:
    """LDS store 192-bit"""
    def __init__(self, dstAddr, src, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStoreB192(self.dstAddr, self.src, comment=self.comment)
        return builder.DSStoreB192(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("DSStoreB256")
class DSStoreB256:
    """LDS store 256-bit"""
    def __init__(self, dstAddr, src, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStoreB256(self.dstAddr, self.src, comment=self.comment)
        return builder.DSStoreB256(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("DSStore2B32")
class DSStore2B32:
    """LDS store 2x32-bit"""
    def __init__(self, dstAddr, src0, src1, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src0 = src0
        self.src1 = src1
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStore2B32(self.dstAddr, self.src0, self.src1, comment=self.comment)
        return builder.DSStore2B32(self.dstAddr, self.src0, self.src1, self.ds, self.comment)

@instruction_wrapper("DSStore2B64")
class DSStore2B64:
    """LDS store 2x64-bit"""
    def __init__(self, dstAddr, src0, src1, ds = None, comment = ""):
        self.dstAddr = dstAddr
        self.src0 = src0
        self.src1 = src1
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSStore2B64(self.dstAddr, self.src0, self.src1, comment=self.comment)
        return builder.DSStore2B64(self.dstAddr, self.src0, self.src1, self.ds, self.comment)

# Special DS Instructions
@instruction_wrapper("DSBPermuteB32")
class DSBPermuteB32:
    """LDS byte permute"""
    def __init__(self, dst, src0, src1, ds = None, comment = ""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        if self.ds is None:
            return builder.DSBPermuteB32(self.dst, self.src0, self.src1, comment=self.comment)
        return builder.DSBPermuteB32(self.dst, self.src0, self.src1, self.ds, self.comment)

# ============================================================================
# Phase 2d: VMEM (Vector Memory) Instructions
# ============================================================================

# BufferLoad Instructions
@instruction_wrapper("BufferLoadB32")
class BufferLoadB32:
    """Buffer load 32-bit"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadB32(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferLoadB64")
class BufferLoadB64:
    """Buffer load 64-bit"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadB64(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferLoadB96")
class BufferLoadB96:
    """Buffer load 96-bit (3 dwords)"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadB96(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferLoadB128")
class BufferLoadB128:
    """Buffer load 128-bit (4 dwords)"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadB128(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferLoadB192")
class BufferLoadB192:
    """Buffer load 192-bit (6 dwords)"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadB192(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferLoadD16B16")
class BufferLoadD16B16:
    """Buffer load D16 format 16-bit (half precision)"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadD16B16(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferLoadD16HIB16")
class BufferLoadD16HIB16:
    """Buffer load D16 format high 16-bit"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadD16HIB16(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferLoadD16U8")
class BufferLoadD16U8:
    """Buffer load D16 format unsigned 8-bit"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadD16U8(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferLoadD16HIU8")
class BufferLoadD16HIU8:
    """Buffer load D16 format high unsigned 8-bit"""
    def __init__(self, dst, vaddr, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferLoadD16HIU8(self.dst, self.vaddr, self.saddr, self.soffset, **kwargs)

# BufferStore Instructions
@instruction_wrapper("BufferStoreB8")
class BufferStoreB8:
    """Buffer store 8-bit"""
    def __init__(self, vaddr, src, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferStoreB8(self.vaddr, self.src, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferStoreB16")
class BufferStoreB16:
    """Buffer store 16-bit"""
    def __init__(self, vaddr, src, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferStoreB16(self.vaddr, self.src, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferStoreB32")
class BufferStoreB32:
    """Buffer store 32-bit"""
    def __init__(self, vaddr, src, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferStoreB32(self.vaddr, self.src, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferStoreB64")
class BufferStoreB64:
    """Buffer store 64-bit"""
    def __init__(self, vaddr, src, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferStoreB64(self.vaddr, self.src, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferStoreB128")
class BufferStoreB128:
    """Buffer store 128-bit"""
    def __init__(self, vaddr, src, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferStoreB128(self.vaddr, self.src, self.saddr, self.soffset, **kwargs)

@instruction_wrapper("BufferStoreD16HIB16")
class BufferStoreD16HIB16:
    """Buffer store D16 format high 16-bit"""
    def __init__(self, vaddr, src, saddr, soffset, offset=0, offen=None, idxen=None, glc=None, slc=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.saddr = saddr
        self.soffset = soffset
        self.offset = offset
        self.offen = offen
        self.idxen = idxen
        self.glc = glc
        self.slc = slc
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.offset != 0:
            kwargs["offset"] = self.offset
        if self.offen is not None:
            kwargs["offen"] = self.offen
        if self.idxen is not None:
            kwargs["idxen"] = self.idxen
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        return builder.BufferStoreD16HIB16(self.vaddr, self.src, self.saddr, self.soffset, **kwargs)

# FlatLoad Instructions
@instruction_wrapper("FlatLoadB32")
class FlatLoadB32:
    """Flat load 32-bit"""
    def __init__(self, dst, vaddr, glc=None, slc=None, saddr=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatLoadB32(self.dst, self.vaddr, **kwargs)

@instruction_wrapper("FlatLoadB64")
class FlatLoadB64:
    """Flat load 64-bit"""
    def __init__(self, dst, vaddr, glc=None, slc=None, saddr=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatLoadB64(self.dst, self.vaddr, **kwargs)

@instruction_wrapper("FlatLoadB128")
class FlatLoadB128:
    """Flat load 128-bit"""
    def __init__(self, dst, vaddr, glc=None, slc=None, saddr=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatLoadB128(self.dst, self.vaddr, **kwargs)

@instruction_wrapper("FlatLoadB192")
class FlatLoadB192:
    """Flat load 192-bit"""
    def __init__(self, dst, vaddr, glc=None, slc=None, saddr=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatLoadB192(self.dst, self.vaddr, **kwargs)

@instruction_wrapper("FlatLoadD16B16")
class FlatLoadD16B16:
    """Flat load D16 format 16-bit"""
    def __init__(self, dst, vaddr, glc=None, slc=None, saddr=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatLoadD16B16(self.dst, self.vaddr, **kwargs)

@instruction_wrapper("FlatLoadD16HIB16")
class FlatLoadD16HIB16:
    """Flat load D16 format high 16-bit"""
    def __init__(self, dst, vaddr, glc=None, slc=None, saddr=None, comment=""):
        self.dst = dst
        self.vaddr = vaddr
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatLoadD16HIB16(self.dst, self.vaddr, **kwargs)

# FlatStore Instructions
@instruction_wrapper("FlatStoreB32")
class FlatStoreB32:
    """Flat store 32-bit"""
    def __init__(self, vaddr, src, glc=None, slc=None, saddr=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatStoreB32(self.vaddr, self.src, **kwargs)

@instruction_wrapper("FlatStoreB64")
class FlatStoreB64:
    """Flat store 64-bit"""
    def __init__(self, vaddr, src, glc=None, slc=None, saddr=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatStoreB64(self.vaddr, self.src, **kwargs)

@instruction_wrapper("FlatStoreB128")
class FlatStoreB128:
    """Flat store 128-bit"""
    def __init__(self, vaddr, src, glc=None, slc=None, saddr=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatStoreB128(self.vaddr, self.src, **kwargs)

@instruction_wrapper("FlatStoreD16B16")
class FlatStoreD16B16:
    """Flat store D16 format 16-bit"""
    def __init__(self, vaddr, src, glc=None, slc=None, saddr=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatStoreD16B16(self.vaddr, self.src, **kwargs)

@instruction_wrapper("FlatStoreD16HIB16")
class FlatStoreD16HIB16:
    """Flat store D16 format high 16-bit"""
    def __init__(self, vaddr, src, glc=None, slc=None, saddr=None, comment=""):
        self.vaddr = vaddr
        self.src = src
        self.glc = glc
        self.slc = slc
        self.saddr = saddr
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.glc is not None:
            kwargs["glc"] = self.glc
        if self.slc is not None:
            kwargs["slc"] = self.slc
        if self.saddr is not None:
            kwargs["saddr"] = self.saddr
        return builder.FlatStoreD16HIB16(self.vaddr, self.src, **kwargs)

# ============================================================================
# Phase 2e: SMEM (Scalar Memory) Instructions
# ============================================================================

@instruction_wrapper("SLoadB64")
class SLoadB64:
    """Scalar load 64-bit"""
    def __init__(self, dst, base, soffset, smem=None, comment=""):
        self.dst = dst
        self.base = base
        self.soffset = soffset
        self.smem = smem
        self.comment = comment

    def create(self, builder):
        kwargs = {"comment": self.comment}
        if self.smem is not None:
            kwargs["smem"] = self.smem
        return builder.SLoadB64(self.dst, self.base, self.soffset, **kwargs)

# ============================================================================
# Phase 2f: MFMA (Matrix Fused Multiply-Add) Instructions
# ============================================================================

@instruction_wrapper("MFMAInstruction")
class MFMAInstruction:
    """Matrix FMA instruction for matrix multiplication"""
    def __init__(self, instType, accType, variant, mfma1k, acc, a, b, acc2=None, neg=False, reuseA=False, reuseB=False, comment=""):
        self.instType = instType
        self.accType = accType
        self.variant = variant
        self.mfma1k = mfma1k
        self.acc = acc
        self.a = a
        self.b = b
        self.acc2 = acc2
        self.neg = neg
        self.reuseA = reuseA
        self.reuseB = reuseB
        self.comment = comment

    def create(self, builder):
        return builder.MFMAInstruction(
            self.instType, self.accType, self.variant, self.mfma1k,
            self.acc, self.a, self.b, self.acc2, self.neg, self.reuseA, self.reuseB, self.comment
        )

@instruction_wrapper("SMFMAInstruction")
class SMFMAInstruction:
    """Sparse Matrix FMA instruction for sparse matrix multiplication"""
    def __init__(self, instType, accType, variant, mfma1k, acc, a, b, metadata, neg=False, comment=""):
        self.instType = instType
        self.accType = accType
        self.variant = variant
        self.mfma1k = mfma1k
        self.acc = acc
        self.a = a
        self.b = b
        self.metadata = metadata
        self.neg = neg
        self.comment = comment

    def create(self, builder):
        return builder.SMFMAInstruction(
            self.instType, self.accType, self.variant, self.mfma1k,
            self.acc, self.a, self.b, self.metadata, self.neg, self.comment
        )

@instruction_wrapper("MXMFMAInstruction")
class MXMFMAInstruction:
    """MX format Matrix FMA instruction for mixed precision matrix multiplication"""
    def __init__(self, instType, accType, mxScaleAType, mxScaleBType, variant, acc, a, b, acc2, mxsa, mxsb, block, reuseA=False, reuseB=False, comment=""):
        self.instType = instType
        self.accType = accType
        self.mxScaleAType = mxScaleAType
        self.mxScaleBType = mxScaleBType
        self.variant = variant
        self.acc = acc
        self.a = a
        self.b = b
        self.acc2 = acc2
        self.mxsa = mxsa
        self.mxsb = mxsb
        self.block = block
        self.reuseA = reuseA
        self.reuseB = reuseB
        self.comment = comment

    def create(self, builder):
        return builder.MXMFMAInstruction(
            self.instType, self.accType, self.mxScaleAType, self.mxScaleBType,
            self.variant, self.acc, self.a, self.b, self.acc2,
            self.mxsa, self.mxsb, self.block, self.reuseA, self.reuseB, self.comment
        )

@instruction_wrapper("VAddCCOU32")
class VAddCCOU32:
    """Vector add with carry-out (64-bit result)"""
    def __init__(self, dst, dst1, src0, src1, src2, comment=""):
        self.dst = dst
        self.dst1 = dst1
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.comment = comment

    def create(self, builder):
        return builder.VAddCCOU32(self.dst, self.dst1, self.src0, self.src1, self.src2, self.comment)

@instruction_wrapper("SCBranchVCCNZ")
class SCBranchVCCNZ:
    """Scalar conditional branch if VCC not zero"""
    def __init__(self, labelName, comment=""):
        self.labelName = labelName
        self.comment = comment

    def create(self, builder):
        return builder.SCBranchVCCNZ(self.labelName, self.comment)

@instruction_wrapper("SCmpLeI32")
class SCmpLeI32:
    """Scalar compare less-than-or-equal signed 32-bit"""
    def __init__(self, src0, src1, comment=""):
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SCmpLeI32(self.src0, self.src1, self.comment)

@instruction_wrapper("SMinU32")
class SMinU32:
    """Scalar unsigned minimum 32-bit"""
    def __init__(self, dst, src0, src1, comment=""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SMinU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SSleep")
class SSleep:
    """Scalar sleep for specified number of cycles"""
    def __init__(self, simm16, comment=""):
        self.simm16 = simm16
        self.comment = comment

    def create(self, builder):
        return builder.SSleep(self.simm16, self.comment)

@instruction_wrapper("SStoreB32")
class SStoreB32:
    """Scalar store 32-bit"""
    def __init__(self, src, base, soffset, smem=None, comment=""):
        self.src = src
        self.base = base
        self.soffset = soffset
        self.smem = smem
        self.comment = comment

    def create(self, builder):
        return builder.SStoreB32(self.src, self.base, self.soffset, self.smem, self.comment)


@instruction_wrapper("DSStoreB8HID16")
class DSStoreB8HID16:
    """Data share store byte with high 16-bit index"""
    def __init__(self, dstAddr, src, ds=None, comment=""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        return builder.DSStoreB8HID16(self.dstAddr, self.src, self.ds, self.comment)

@instruction_wrapper("BufferAtomicAddF32")
class BufferAtomicAddF32:
    """Buffer atomic add F32"""
    def __init__(self, src, vaddr, saddr, soffset, mubuf=None, comment=""):
        self.src = src
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.mubuf = mubuf
        self.comment = comment

    def create(self, builder):
        return builder.BufferAtomicAddF32(
            self.src, self.vaddr, self.saddr, self.soffset,
            self.mubuf, self.comment
        )

@instruction_wrapper("BufferAtomicCmpswapB32")
class BufferAtomicCmpswapB32:
    """Buffer atomic compare-and-swap 32-bit"""
    def __init__(self, src, vaddr, saddr, soffset, mubuf=None, comment=""):
        self.src = src
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.mubuf = mubuf
        self.comment = comment

    def create(self, builder):
        return builder.BufferAtomicCmpswapB32(
            self.src, self.vaddr, self.saddr, self.soffset,
            self.mubuf, self.comment
        )

@instruction_wrapper("TensorLoadToLds")
class TensorLoadToLds:
    """Tensor load to LDS"""
    def __init__(self, group0, group1, group2=None, group3=None, comment=""):
        self.group0 = group0
        self.group1 = group1
        self.group2 = group2
        self.group3 = group3
        self.comment = comment

    def create(self, builder):
        return builder.TensorLoadToLds(self.group0, self.group1, self.group2, self.group3, self.comment)

@instruction_wrapper("SMulHIU32")
class SMulHIU32:
    """Scalar multiply high unsigned 32-bit"""
    def __init__(self, dst, src0, src1, comment=""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SMulHIU32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SAtomicDec")
class SAtomicDec:
    """Scalar atomic decrement"""
    def __init__(self, dst, base, smem=None, comment=""):
        self.dst = dst
        self.base = base
        self.smem = smem
        self.comment = comment

    def create(self, builder):
        return builder.SAtomicDec(self.dst, self.base, self.smem, self.comment)

@instruction_wrapper("SSubI32")
class SSubI32:
    """Scalar subtract signed 32-bit"""
    def __init__(self, dst, src0, src1, comment=""):
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.comment = comment

    def create(self, builder):
        return builder.SSubI32(self.dst, self.src0, self.src1, self.comment)

@instruction_wrapper("SCBranchExecNZ")
class SCBranchExecNZ:
    """Scalar conditional branch if EXEC not zero"""
    def __init__(self, labelName, comment=""):
        self.labelName = labelName
        self.comment = comment

    def create(self, builder):
        return builder.SCBranchExecNZ(self.labelName, self.comment)

@instruction_wrapper("SCBranchExecZ")
class SCBranchExecZ:
    """Scalar conditional branch if EXEC is zero"""
    def __init__(self, labelName, comment=""):
        self.labelName = labelName
        self.comment = comment

    def create(self, builder):
        return builder.SCBranchExecZ(self.labelName, self.comment)

@instruction_wrapper("SSwapPCB64")
class SSwapPCB64:
    """Scalar swap PC 64-bit"""
    def __init__(self, dst, src, comment=""):
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        return builder.SSwapPCB64(self.dst, self.src, self.comment)

@instruction_wrapper("BufferAtomicCmpswapB64")
class BufferAtomicCmpswapB64:
    """Buffer atomic compare-and-swap 64-bit"""
    def __init__(self, src, vaddr, saddr, soffset, mubuf=None, comment=""):
        self.src = src
        self.vaddr = vaddr
        self.saddr = saddr
        self.soffset = soffset
        self.mubuf = mubuf
        self.comment = comment

    def create(self, builder):
        return builder.BufferAtomicCmpswapB64(
            self.src, self.vaddr, self.saddr, self.soffset,
            self.mubuf, self.comment
        )

@instruction_wrapper("FlatAtomicCmpswapB32")
class FlatAtomicCmpswapB32:
    """Flat atomic compare-and-swap 32-bit"""
    def __init__(self, vaddr, tmp, src, flat=None, comment=""):
        self.vaddr = vaddr
        self.tmp = tmp
        self.src = src
        self.flat = flat
        self.comment = comment

    def create(self, builder):
        return builder.FlatAtomicCmpswapB32(self.vaddr, self.tmp, self.src, self.flat, self.comment)

@instruction_wrapper("DSLoadU8")
class DSLoadU8:
    """Data share load unsigned 8-bit"""
    def __init__(self, dst, src, ds=None, comment=""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        return builder.DSLoadU8(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadU16")
class DSLoadU16:
    """Data share load unsigned 16-bit"""
    def __init__(self, dst, src, ds=None, comment=""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        return builder.DSLoadU16(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadD16HIU8")
class DSLoadD16HIU8:
    """Data share load high 16-bit from unsigned 8-bit"""
    def __init__(self, dst, src, ds=None, comment=""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        return builder.DSLoadD16HIU8(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSLoadD16HIU16")
class DSLoadD16HIU16:
    """Data share load high 16-bit from unsigned 16-bit"""
    def __init__(self, dst, src, ds=None, comment=""):
        self.dst = dst
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        return builder.DSLoadD16HIU16(self.dst, self.src, self.ds, self.comment)

@instruction_wrapper("DSStoreD16HIB16")
class DSStoreD16HIB16:
    """Data share store high 16-bit as 16-bit"""
    def __init__(self, dstAddr, src, ds=None, comment=""):
        self.dstAddr = dstAddr
        self.src = src
        self.ds = ds
        self.comment = comment

    def create(self, builder):
        return builder.DSStoreD16HIB16(self.dstAddr, self.src, self.ds, self.comment)

