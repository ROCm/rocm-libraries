################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

"""
A component is a piece of code that is chosen among compatible components based on the current hardware capabilities and/or kernel options.

The class hierarchy is automatically used to detect which type of component a given class belongs to. For example, all the MAC components should inherit from the MAC class.

Most components should be able to get away with defining their requirements via the class properties (e.g.):

```python
    class FMA_NonPacked(MAC):
        asmCaps = {"v_fma_f16": True,
                "v_pk_fma_f16": False}
        #archCaps = {}
        kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                                "HighPrecisionAccumulate": False}}
```

Values in the dictionaries can be lambdas for more advanced logic:

```python
    class FMA_HPA_MAD_MIX(MAC):
        asmCaps = {"v_mad_mix_f32": True}
        #archCaps = {}
        kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                                "HighPrecisionAccumulate": True},
                }
```

Any more advanced logic should be implemented by overriding the matches() method.

Components are found by calling `Component.<subtype>.find(writer)` where `writer` is a `KernelWriter` object:

```python
    component = Component.MAC.find(self)
    if component:
      return component(self, m, innerUnroll)

    # No component was found, fall back to existing code...
```

With this fallback mechanism, components can be added one at a time, without disrupting existing code.

Components can be categorized in different files in the `Tensile/Components` directory.  Each file should be listed in the `__all__` member of `Tensile/Components/__init__.py`.
"""

import abc
from collections.abc import Mapping
import inspect
from dataclasses import dataclass

@dataclass
class LraTileProperties:
    """
    Lra tile assignment properties.
    """
    pass

def PartialMatch(pattern, obj, debug=False, level=0):
    indent = "    " * level
    if debug and level == 0:
        print("")
    if hasattr(pattern, "__call__"):
        if not pattern(obj):
            if debug:
                print("{indent}call({obj}) == False".format(indent=indent, obj=obj))
            return False
    elif isinstance(pattern, Mapping) and \
         isinstance(obj, Mapping):
        for key, value in pattern.items():
            if key not in obj:
                if debug:
                    print("{indent}{key} not in object.".format(indent=indent, key=key))
                return False

            if debug:
                print("{indent} recursing into {key}".format(indent=indent, key=key))
            if not PartialMatch(value, obj[key], debug, level+1):
                return False

    elif pattern != obj:
        if debug:
            print("{indent}{pattern} != {obj}".format(indent=indent, pattern=pattern, obj=obj))
        return False

    if debug:
        print("{indent}: True".format(indent=indent))
    return True

class ComponentMeta(abc.ABCMeta):
    """
    Metaclass which auto-registers each subclass in an "implementations"
    member of its parent class, to allow for hierarchical searching.
    """
    def __init__(cls, name, bases, namespace, **kwargs):
        if inspect.isabstract(cls):
            cls.implementations = {}

        for base in bases:
            base.implementations[name] = cls
            setattr(base, name, cls)

class Component(metaclass=ComponentMeta):
    """
    Modular component which allows kernel components to be specified based on
    capability rather than based on individual architecture IDs.
    """

    @classmethod
    def matches(cls, writer, debug=False):
        if hasattr(cls, "versions"):
            if not writer.version in cls.versions:
                return False

        attrs = ["asmCaps", "archCaps", "kernel"]
        for attr in attrs:
            if hasattr(cls, attr):
                if not PartialMatch(getattr(cls, attr), getattr(writer.states, attr), debug):
                    return False

        return True

    @classmethod
    def findAll(cls, writer, debug=False, *args, **kwargs):
        found = []
        for name, impl in cls.implementations.items():
            if debug:
                print(name)
            if inspect.isabstract(impl):
                foundHere = impl.findAll(writer, debug, *args, **kwargs)
                if debug:
                    print(name, ": found ", foundHere)
                found += foundHere
            elif impl.matches(writer, debug, *args, **kwargs):
                if debug:
                    print(name, ": found impl")
                found.append(impl)
            else:
                if debug:
                    print(name, "mismatch!")

        return found

    @classmethod
    def find(cls, writer, debug=False, *args, **kwargs):
        found = cls.findAll(writer, debug, *args, **kwargs)

        if len(found) == 0:
            return None

        if len(found) > 1:
            raise RuntimeError("Found {} implementations for {}".format(len(found), cls.__name__))

        return found[0]()

    @classmethod
    def componentPath(cls, path=None, bases=None):
        if path is None:
            path = []

        if bases is None:
            bases = cls.__bases__

        if not isinstance(cls, str):
            className = cls.__name__
        path = [className] + path

        if cls == Component or len(bases) == 0:
            return path
        return bases[0].componentPath(path)

    @abc.abstractmethod
    def __call__(self):
        """
        Concrete subclasses must implement __call__.
        """
        pass

    def commentHeader(self):
        """
        Returns a comment which helps identify where a piece of code was generated.
        """
        return "{}".format('.'.join(self.componentPath()))

class MAC(Component):
    """
    Multiply-accumulate block.
    """
    pass

class Signature(Component):
    """
    Function signature block.
    """
    pass

class LocalRead(Component):
    """
    Local read block.
    """
    pass

class SumUnroll(Component):
    """
    Sum unroll block.
    """
    @abc.abstractmethod
    def initSumUnroll(self, writer, kernel):
        pass

    @abc.abstractmethod
    def loopSum(self, writer, kernel, tP, u, innerUnroll):
        pass

    @abc.abstractmethod
    def storeSumLDS(self, writer, kernel, tP):
        pass

class ShiftVectorComponents(Component):
    """
    Shift vector components block.
    """
    pass

class ComputeStoreVgprs(Component):
    """
    Compute store vgprs block.
    """
    pass

class NotLocalFullTileElements(Component):
    """
    Not local full tile elements block.
    """
    pass

class LraTileAssignment(Component):
    """
    Lra tile assignment block.
    """
    pass

class PackData(Component):
    """
    Pack data block.
    """
    pass

class SIA(Component):
    """
    ScheduleIterAlg block.
    """
    @abc.abstractmethod
    def schedIntoIteration(self, writer, kernel, tensorParametersA, tensorParametersB, \
        localWriteEndIter, firstIter, lastLoop, lastLc, maxVmcnt, globalReadIncACode, \
        globalReadIncBCode):
        pass

class GlobalWriteComponents(Component):
    pass

# Importing here allows auto-registry of components in the Components directory.
# Each file must be listed in __all__ in Components/__init__.py
# "noqa" prevents linter from complaining here.
from .Components import *  # noqa
