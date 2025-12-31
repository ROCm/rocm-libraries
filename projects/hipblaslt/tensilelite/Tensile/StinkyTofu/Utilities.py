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

from . import ModuleAdapter

def isinstance_item(obj, class_or_tuple):
    """
    Custom isinstance that works with both wrapper and rocisa instances.

    Handles these cases:
    - isinstance_item(wrapped_instance, WrapperClass) -> True
    - isinstance_item(wrapped_instance, rocisa_class) -> True (checks _rocisa_inst)
    - isinstance_item(rocisa_instance, WrapperClass) -> True (checks _rocisa_class)
    - isinstance_item(rocisa_instance, rocisa_class) -> True

    Usage:
        from Tensile.StinkyTofu.ItemAdapters import isinstance_item, SNop

        # Works with both wrapper and rocisa instances
        if isinstance_item(item, SNop):
            pass
    """
    # Handle tuple of classes
    if isinstance(class_or_tuple, tuple):
        return any(isinstance_item(obj, cls) for cls in class_or_tuple)

    cls = class_or_tuple

    # Standard isinstance check first
    if isinstance(obj, cls):
        return True

    # Check if obj is a wrapper with _rocisa_inst matching the class
    if hasattr(obj, '_rocisa_inst') and obj._rocisa_inst is not None:
        if isinstance(obj._rocisa_inst, cls):
            return True

    # Check if obj is rocisa instance and cls is wrapper with matching _rocisa_class
    if hasattr(cls, '_rocisa_class'):
        if isinstance(obj, cls._rocisa_class):
            return True

    if ModuleAdapter.rocisa_binding and cls in (ModuleAdapter.rocisa_code.Module, ModuleAdapter.Module):
        return isinstance(obj, (ModuleAdapter.rocisa_code.Module, ModuleAdapter.Module))

    return False


import rocisa.container as rocisa_container
import rocisa

def replaceHolder(item, dst):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa_container.replaceHolder(item.rocisa(), dst)
    elif hasattr(item, '_rocisa_inst'):
        return rocisa_container.replaceHolder(item._rocisa_inst, dst)
    else:
        return rocisa_container.replaceHolder(item, dst)

def countInstruction(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countInstruction(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countInstruction(item._rocisa_inst)
    else:
        return rocisa.countInstruction(item)

def countGlobalRead(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countGlobalRead(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countGlobalRead(item._rocisa_inst)
    else:
        return rocisa.countGlobalRead(item)

def countLocalRead(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countLocalRead(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countLocalRead(item._rocisa_inst)
    else:
        return rocisa.countLocalRead(item)

def countLocalWrite(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countLocalWrite(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countLocalWrite(item._rocisa_inst)
    else:
        return rocisa.countLocalWrite(item)

def countWeightedLocalRead(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countWeightedLocalRead(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countWeightedLocalRead(item._rocisa_inst)
    else:
        return rocisa.countWeightedLocalRead(item)

def countWeightedLocalWrite(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countWeightedLocalWrite(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countWeightedLocalWrite(item._rocisa_inst)
    else:
        return rocisa.countWeightedLocalWrite(item)

def getMFMAs(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.getMFMAs(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.getMFMAs(item._rocisa_inst)
    else:
        return rocisa.getMFMAs(item)

def countSMemLoad(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countSMemLoad(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countSMemLoad(item._rocisa_inst)
    else:
        return rocisa.countSMemLoad(item)

def findInstCount(item, target_item, count):
    if isinstance(item, ModuleAdapter.Module):
        if hasattr(item, '_rocisa_inst'):
            return rocisa.findInstCount(item.rocisa(), target_item._rocisa_inst, count)
        else:
            return rocisa.findInstCount(item.rocisa(), target_item, count)
    else:
        if hasattr(target_item, '_rocisa_inst'):
            return rocisa.findInstCount(item, target_item._rocisa_inst, count)
        else:
            return rocisa.findInstCount(item, target_item, count)

def countDSStoreB128(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countDSStoreB128(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countDSStoreB128(item._rocisa_inst)
    else:
        return rocisa.countDSStoreB128(item)

def countDSStoreB192(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countDSStoreB192(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countDSStoreB192(item._rocisa_inst)
    else:
        return rocisa.countDSStoreB192(item)

def countVMovB32(item):
    if isinstance(item, ModuleAdapter.Module):
        return rocisa.countVMovB32(item.rocisa())
    elif hasattr(item, '_rocisa_inst'):
        return rocisa.countVMovB32(item._rocisa_inst)
    else:
        return rocisa.countVMovB32(item)