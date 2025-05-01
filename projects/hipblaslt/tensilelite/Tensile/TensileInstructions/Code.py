################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from ..Common.Architectures import isaToGfx
from .Base import Item
from .Enums import SignatureValueKind
from .Formatting import slash, slash50, block, block3Line, blockNewLine, \
                        formatStr, printExit
from .Instructions import Instruction, MacroInstruction

from math import ceil
from typing import Optional
import ctypes

# Global to print module names around strings
printModuleNames = 0

class Label (Item):
    """
    Label that can be the target of a jump.
    """
    def __init__(self, label, comment):
        super().__init__("")
        assert(isinstance(label, int) or isinstance(label, str))
        self.label = label
        self.comment = comment

    @staticmethod
    def getFormatting(label):
        if isinstance(label, int):
            return "label_%04u" % (label)
        else:
            return "label_%s" % (label)

    def getLabelName(self):
        return Label.getFormatting(self.label)

    def __str__(self):
        t = self.getLabelName() + ":"
        if self.comment:
            t += "  /// %s" % self.comment
        t += "\n"
        return t

class Macro(Item):
    def __init__(self, *args):
        super().__init__("")
        self.itemList = []
        self.macro = ""
        self.addTitle(*args)

    def addTitle(self, *args):
        self.name = args[0]
        self.macro = MacroInstruction(name=args[0], args=list(args[1:]))

    def add(self, item):
        # This is a workaround
        if isinstance(item, (Instruction, Module, TextBlock)):
            item.parent = self # type: ignore
            self.itemList.append(item)
        else:
            assert 0, "unknown item type (%s) for Code.add. item=%s"%(type(item), item)
        return item

    def addComment0(self, comment):
        """
        Convenience function to format arg as a comment and add TextBlock item
        This comment is a single line /* MYCOMMENT  */
        """
        self.add(TextBlock("/* %s */\n"%comment))

    def addselfAsm(self, comment):
        self.add(TextBlock(comment))

    def prettyPrint(self,indent=""):
        ostream = ""
        ostream += '%s%s "%s"\n'%(indent, type(self).__name__, self.name)
        for i in self.itemList:
            ostream += i.prettyPrint(indent.replace("|--", "| ") + "|--")
        return ostream

    def setItems(self, itemList):
        self.itemList = itemList

    def items(self):
        """
        Return list of items in the Macro
        Items may be other Inst
        """
        return self.itemList

    def __str__(self):
        assert(self.macro)
        s = ""
        if printModuleNames:
            s += "// %s { \n" % self.name
        s += ".macro " + str(self.macro).replace(",", "")
        s += "".join([("    " + str(x).replace(",", "")) for x in self.itemList])
        s += ".endm\n"
        if printModuleNames:
            s += "// } %s\n" % self.name
        return s

class Module(Item):
    """
    Modules contain lists of text instructions, Inst objects, or additional modules
    They can be easily converted to string that represents all items in the list
    and can be mixed with standard text.
    The intent is to allow the kernel writer to express the structure of the
    code (ie which instructions are a related module) so the scheduler can later
    make intelligent and legal transformations.
    """
    def __init__(self, name="") -> None:
        super().__init__(name)
        self.itemList = []
        self.tempVgpr = None
        self._isNoOpt = False

    def setNoOpt(self, noOpt: bool) -> None:
        self._isNoOpt = noOpt

    def isNoOpt(self) -> bool:
        return self._isNoOpt

    def findNamedItem(self, targetName):
        return next((item for item in self.itemList if item.name==targetName), None)

    def setInlineAsmPrintMode(self, mode):
        for item in self.itemList:
            if isinstance(item, Module):
                item.setInlineAsmPrintMode(mode)
            elif isinstance(item, Instruction):
                item.setInlineAsm(mode)

    def __str__(self):
        prefix = f"// {self.name}{{\n" if printModuleNames else ""
        suffix = f"// }} {self.name}\n" if printModuleNames else ""
        s = "".join(str(x) for x in self.itemList)
        return "".join((prefix, s, suffix))

    def addSpaceLine(self):
        self.itemList.append(TextBlock("\n"))

    def add(self, item, pos=-1):
        """
        Add specified item to the list of items in the module.
        Item MUST be a Item (not a string) - can use
        addText(...)) to add a string.
        All additions to itemList should use this function.

        Returns item to facilitate one-line create/add patterns
        """
        if isinstance(item, Item):
            item.parent = self # type: ignore
            if pos == -1:
                self.itemList.append(item)
            else:
                self.itemList.insert(pos, item)
        else:
            assert 0, "unknown item type (%s) for Module.add. item=%s"%(type(item), item)
        return item

    def appendModule(self, module):
        """
        Append items to module.
        """
        assert(isinstance(module, Module))
        for i in module.items():
            self.add(i)
        return module

    def addModuleAsFlatItems(self, module):
        """
        Add items to module.

        Returns items to facilitate one-line create/add patterns
        """
        assert(isinstance(module, Module))
        for i in module.flatitems():
            self.add(i)
        return module

    def findIndex(self, targetItem):
        if isinstance(targetItem, Item):
            return self.itemList.index(targetItem)
        return -1

    def findIndexByType(self, targetType):
        for i, item in enumerate(self.itemList):
            if isinstance(item, targetType):
                return i
        return None

    def addComment(self, comment):
        """
        Convenience function to format arg as a comment and add TextBlock item
        This comment is a single line // MYCOMMENT
        """
        self.add(TextBlock(slash(comment)))

    def addCommentAlign(self, comment):
        """
        Convenience function to format arg as a comment and add TextBlock item
        This comment is a single line // MYCOMMENT with the same format as
        Instruction.
        """
        self.add(TextBlock(slash50(comment)))

    def addComment0(self, comment):
        """
        Convenience function to format arg as a comment and add TextBlock item
        This comment is a single line /* MYCOMMENT  */
        """
        self.add(TextBlock(block(comment)))

    def addComment1(self, comment):
        """
        Convenience function to format arg as a comment and add TextBlock item
        This comment is a blank line followed by /* MYCOMMENT  */
        """
        self.add(TextBlock(blockNewLine(comment)))

    def addComment2(self, comment):
        self.add(TextBlock(block3Line(comment)))

    def addselfAsm(self, comment):
        self.add(TextBlock(comment))

    def prettyPrint(self,indent=""):
        ostream = ""
        ostream += '%s%s "%s"\n'%(indent, type(self).__name__, self.name)
        for i in self.itemList:
            ostream += i.prettyPrint(indent.replace("|--", "| ") + "|--")
        return ostream
        """
        Test code:
          mod1 = Code.Module("TopModule")
          mod2 = Code.Module("Module-lvl2")
          mod2.add(Code.Inst("bogusInst", "comments"))
          mod3 = Code.Module("Module-lvl3")
          mod3.add(Code.TextBlock("bogusTextBlock\nbogusTextBlock2\nbogusTextBlock3"))
          mod3.add(Code.GlobalReadInst("bogusGlobalReadInst", "comments"))
          mod2.add(mod3)
          mod1.add(mod2)
          mod1.add(Code.Inst("bogusInst", "comments"))
          mod1.add(mod2)

          print(mod1.prettyPrint())
        Output:
          Module "TopModule"
          |--Module "Module-lvl2"
          | |--Inst bogusInst                                          // comments
          | |--Module "Module-lvl3"
          | | |--TextBlock
          | | | |--bogusTextBlock
          | | | |--bogusTextBlock2
          | | | |--bogusTextBlock3
          | | |--GlobalReadInst bogusGlobalReadInst                                // comments
          |--Inst bogusInst                                          // comments
          |--Module "Module-lvl2"
          | |--Inst bogusInst                                          // comments
          | |--Module "Module-lvl3"
          | | |--TextBlock
          | | | |--bogusTextBlock
          | | | |--bogusTextBlock2
          | | | |--bogusTextBlock3
          | | |--GlobalReadInst bogusGlobalReadInst                                // comments
        """

    def countTypeList(self, ttypeList):
        count = 0
        # add "Module" type to type list filter, where we want to count recursively
        # the types under "Module"
        if Module not in ttypeList:
            ttypeList.append(Module)
        for ttype in ttypeList:
            count += self.countType(ttype)
        return count

    def countType(self,ttype):
        """
        Count number of items with specified type in this Module
        Will recursively count occurrences in submodules
        (Overrides Item.countType)
        """
        count=0
        for i in self.itemList:
            count += i.countType(ttype)
        return count

    def count(self):
        count=0
        for i in self.itemList:
            if isinstance(i, Module):
                count += i.count()
            else:
                count += 1
        return count

    def setItems(self, itemList):
        self.itemList = itemList

    def items(self):
        """
        Return list of items in the Module
        Items may be other Modules, TexBlock, or Inst
        """
        return self.itemList

    def replaceItem(self, srcItem, dstItem):
        """
        Replace item from itemList.
        Items may be other Modules, TexBlock, or Inst
        """
        for index, s in enumerate(self.itemList):
            if s is srcItem:
                dstItem.parent = self
                self.itemList[index] = dstItem
                break

    def replaceItemByIndex(self, index, item):
        """
        Replace item from itemList, do nothing if
        exceed length of the itemList
        Items may be other Modules, TexBlock, or Inst
        """
        if index >= len(self.itemList):
            return
        item.parent = self
        self.itemList[index] = item

    def removeItemByIndex(self, index):
        """
        Remove item from itemList, remove the last element if
        exceed length of the itemList
        Items may be other Modules, TexBlock, or Inst
        """
        if index >= len(self.itemList):
            index = -1
        del self.itemList[index]

    def removeItem(self, item):
        self.itemList = [ x for x in self.itemList if x is not item ]

    def removeItemsByName(self, name):
        """
        Remove items from itemList
        Items may be other Modules, TexBlock, or Inst
        """
        self.itemList = [ x for x in self.itemList if x.name != name ]

    def flatitems(self):
        """
        Return flattened list of items in the Module
        Items in sub-modules will be flattened into single list
        Items may be TexBlock or Inst
        """
        flatitems = []
        for i in self.itemList:
            if isinstance(i, Module):
                flatitems += i.flatitems()
            else:
                flatitems.append(i)
        return flatitems

    def addTempVgpr(self, vgpr):
        self.tempVgpr = vgpr

class StructuredModule(Module):
    def __init__(self, name=""):
        Module.__init__(self, name)
        self.header = Module("header")
        self.middle = Module("middle")
        self.footer =  Module("footer")

        self.add(self.header)
        self.add(self.middle)
        self.add(self.footer)

class TextBlock(Item):
    """
    An unstructured block of text
    """
    def __init__(self, text: str):
        super().__init__(text)
        self.text = text

    def __str__(self) -> str:
        return self.text

class ValueEndif(Item):
    def __init__(self, comment=""):
        super().__init__("ValueEndif")
        self.comment = comment

    def __str__(self):
        return formatStr(False, ".endif", self.comment)

class ValueIf(Item):
    def __init__(self, value: int):
        super().__init__("ValueIf")
        self.value = value

    def __str__(self):
        return ".if " + str(self.value)

class ValueSet(Item):
    def __init__(self, name, value, offset = 0, format = 0):
        super().__init__(name)
        if isinstance(value, int):
            self.ref   = None
            self.value = value
        elif isinstance(value, str):
            self.ref   = value
            self.value = None
        self.offset = offset
        # -1 for no offset, 0 for dec, 1 for hex
        self.format = format

    def __str__(self):
        t = ".set " + self.name + ", "
        if self.ref != None:
            if self.format == -1:
                t += str(self.ref)
            else:
                t += "%s+%u" % (self.ref, self.offset)
        elif self.value != None:
            if self.format == -1:
                t += str(self.value)
            elif self.format == 0:
                t += str(self.value + self.offset)
            elif self.format == 1:
                t += "0x{0:08x}".format(self.value + self.offset)
        t += "\n"
        return t

class RegSet(ValueSet):
    def __init__(self, regType, name, value, offset = 0):
        super().__init__(name, value, offset)
        # v or s
        self.regType = regType

class BitfieldStructure(ctypes.Structure):
    def field_desc(self, field):
        fname = field[0]
        bits = " ({}b)".format(field[2]) if len(field) > 2 else ""
        value = getattr(self, fname)
        return "{0}{1}: {2}".format(fname, bits, value)

    def desc(self):
        return '\n'.join([self.field_desc(field) for field in self._fields_])

class BitfieldUnion(ctypes.Union):
    def __str__(self):
        return "0x{0:08x}".format(self.value)

    def getValue(self):
        return self.value

    def desc(self):
        return "hex: {}\n".format(self) + self.fields.desc()

class SrdUpperFields9XX(BitfieldStructure): #0x00020000
    _fields_ = [("dst_sel_x",      ctypes.c_uint, 3),
                ("dst_sel_y",      ctypes.c_uint, 3),
                ("dst_sel_z",      ctypes.c_uint, 3),
                ("dst_sel_w",      ctypes.c_uint, 3),
                ("num_format",     ctypes.c_uint, 3),
                ("data_format",    ctypes.c_uint, 4),#111-113
                ("user_vm_enable", ctypes.c_uint, 1),
                ("user_vm_mode",   ctypes.c_uint, 1),
                ("index_stride",   ctypes.c_uint, 2),
                ("add_tid_enable", ctypes.c_uint, 1),
                ("_unusedA",       ctypes.c_uint, 3),
                ("nv",             ctypes.c_uint, 1),
                ("_unusedB",       ctypes.c_uint, 2),
                ("type",           ctypes.c_uint, 2)]

    @classmethod
    def default(cls):
        return cls(data_format = 4)

class SrdUpperValue9XX(BitfieldUnion):
    _fields_ = [("fields", SrdUpperFields9XX), ("value", ctypes.c_uint32)]

    @classmethod
    def default(cls):
        return cls(fields=SrdUpperFields9XX.default())

class SrdUpperFields10XX(BitfieldStructure):
    _fields_ = [("dst_sel_x",      ctypes.c_uint, 3),
                ("dst_sel_y",      ctypes.c_uint, 3),
                ("dst_sel_z",      ctypes.c_uint, 3),
                ("dst_sel_w",      ctypes.c_uint, 3),
                ("format",         ctypes.c_uint, 7),
                ("_unusedA",       ctypes.c_uint, 2),
                ("index_stride",   ctypes.c_uint, 2),
                ("add_tid_enable", ctypes.c_uint, 1),
                ("resource_level", ctypes.c_uint, 1),
                ("_unusedB",       ctypes.c_uint, 1),
                ("LLC_noalloc",    ctypes.c_uint, 2),
                ("oob_select",     ctypes.c_uint, 2),
                ("type",           ctypes.c_uint, 2)]


    @classmethod
    def default(cls):
        return cls(format         = 4,
                   resource_level = 1,
                   oob_select     = 3)


class SrdUpperValue10XX(BitfieldUnion):
    _fields_ = [("fields", SrdUpperFields10XX), ("value", ctypes.c_uint32)]

    @classmethod
    def default(cls):
        return cls(fields=SrdUpperFields10XX.default())


class SrdUpperFields11XX(BitfieldStructure): #0x31004000
    _fields_ = [("dst_sel_x",      ctypes.c_uint, 3),
                ("dst_sel_y",      ctypes.c_uint, 3),
                ("dst_sel_z",      ctypes.c_uint, 3),
                ("dst_sel_w",      ctypes.c_uint, 3),
                ("format",         ctypes.c_uint, 7), #108-114
                ("_unusedA",       ctypes.c_uint, 2),
                ("index_stride",   ctypes.c_uint, 2), #117-118
                ("add_tid_enable", ctypes.c_uint, 1), #119
                ("resource_level", ctypes.c_uint, 1), #120
                ("_unusedB",       ctypes.c_uint, 1), #121
                ("LLC_noalloc",    ctypes.c_uint, 2), #122-123
                ("oob_select",     ctypes.c_uint, 2), #124-125
                ("type",           ctypes.c_uint, 2)] #126-127

    @classmethod
    def default(cls):
        return cls(format         = 4,
                   resource_level = 1,
                   oob_select     = 3)

class SrdUpperValue11XX(BitfieldUnion):
    _fields_ = [("fields", SrdUpperFields11XX), ("value", ctypes.c_uint32)]

    @classmethod
    def default(cls):
        return cls(fields=SrdUpperFields11XX.default())

class SrdUpperFields12XX(BitfieldStructure): #0x10020000
    _fields_ = [("dst_sel_x",      ctypes.c_uint, 3), #96-
                ("dst_sel_y",      ctypes.c_uint, 3),
                ("dst_sel_z",      ctypes.c_uint, 3),
                ("dst_sel_w",      ctypes.c_uint, 3),
                ("format",         ctypes.c_uint, 7), #108-114
                ("_unusedA",       ctypes.c_uint, 2),
                ("index_stride",   ctypes.c_uint, 2), #117-118
                ("add_tid_enable", ctypes.c_uint, 1), #119
                ("resource_level", ctypes.c_uint, 1), #120
                ("_unusedB",       ctypes.c_uint, 3), #121-123
                #("LLC_noalloc",    ctypes.c_uint, 2), #122-123
                ("oob_select",     ctypes.c_uint, 2), #124-125
                ("type",           ctypes.c_uint, 2)] #126-127

    @classmethod
    def default(cls):
        return cls(format         = 32,
                   #resource_level = 1,
                   oob_select     = 3)

class SrdUpperValue12XX(BitfieldUnion):
    _fields_ = [("fields", SrdUpperFields12XX), ("value", ctypes.c_uint32)]

    @classmethod
    def default(cls):
        return cls(fields=SrdUpperFields12XX.default())

def SrdUpperValue(isa):
    if isa[0] == 12:
        return SrdUpperValue12XX.default()
    elif isa[0] == 11:
        return SrdUpperValue11XX.default()
    elif isa[0] == 10:
        return SrdUpperValue10XX.default()
    else:
        return SrdUpperValue9XX.default()

########################################
# Signatures
########################################

class _SignatureArgument(Item):

    ValueTypeSizeDict = {'i8':  1,
                         'i16': 2,
                         'i32': 4,
                         'i64': 8,
                         'u8':  1,
                         'u16': 2,
                         'u32': 4,
                         'u64': 8,
                         'bf16': 2,
                         'f16': 2,
                         'f32': 4,
                         'f64': 8,
                         'pkf16': 4,
                         'struct': 8
                        }

    def __init__(self, name, valueKind, valueType, addrSpaceQual = None):
        super().__init__(name)
        self.valueKind = valueKind
        self.valueType = valueType
        self.size      = self.valueToSize(valueKind, valueType)

        self.addrSpaceQual = addrSpaceQual

    def valueToSize(self, valueKind, valueType):
        if valueKind == SignatureValueKind.SIG_GLOBALBUFFER:
            return 8

        return _SignatureArgument.ValueTypeSizeDict[valueType]

    def valueKindToStr(self):
        if self.valueKind == SignatureValueKind.SIG_GLOBALBUFFER:
            return "global_buffer"
        elif self.valueKind == SignatureValueKind.SIG_VALUE:
            return "by_value"

class _SignatureArgument(_SignatureArgument):
    def __init__(self, offset, name, valueKind, valueType, addrSpaceQual=None):
        super().__init__(name, valueKind, valueType, addrSpaceQual)
        self.offset = offset

    def __str__(self):
        signatureIndent = " " * 8
        kStr = ""
        kStr += signatureIndent[2:] + "- .name:            %s\n" % self.name
        kStr += signatureIndent + ".size:            %s\n" % self.size
        kStr += signatureIndent + ".offset:          %s\n" % self.offset
        kStr += signatureIndent + ".value_kind:      %s\n" % self.valueKindToStr()
        kStr += signatureIndent + ".value_type:      %s\n" % self.valueType
        if self.addrSpaceQual != None:
            kStr += signatureIndent + ".address_space:   %s\n" % self.addrSpaceQual
        return kStr

class _SignatureKernelDescriptor(Item):
    def __init__(self, name, groupSegSize, sgprWorkGroup, vgprWorkItem, \
        totalVgprs: int=0, totalAgprs: int=0, totalSgprs: int =0, preloadKernArgs: bool=False):
        super().__init__(name)
        # accumulator offset for Unified Register Files
        if self.archCaps["ArchAccUnifiedRegs"]:
            self.accumOffset = ceil(totalVgprs/8)*8
            self.totalVgprs = self.accumOffset + totalAgprs
        else:
            self.accumOffset = None
            self.totalVgprs = totalVgprs
        self.originalTotalVgprs = totalVgprs
        self.totalAgprs         = totalAgprs
        self.totalSgprs         = totalSgprs
        self.groupSegSize = groupSegSize
        self.sgprWorkGroup = sgprWorkGroup
        self.vgprWorkItem = vgprWorkItem
        self.enablePreloadKernArgs = preloadKernArgs

    def setGprs(self, totalVgprs: int, totalAgprs: int, totalSgprs: int):
        if self.archCaps["ArchAccUnifiedRegs"]:
            self.accumOffset = ceil(totalVgprs/8)*8
            self.totalVgprs = self.accumOffset + totalAgprs
        else:
            self.accumOffset = None
            self.totalVgprs = max(totalAgprs, totalVgprs)
        self.originalTotalVgprs = totalVgprs
        self.totalAgprs         = totalAgprs
        self.totalSgprs         = totalSgprs

    def getNextFreeVgpr(self) -> int:
        return self.totalVgprs

    def getNextFreeSgpr(self) -> int:
        return self.totalSgprs

    def __str__(self):
        kdIndent = " " * 2
        kStr = ""
        kStr += ".amdgcn_target \"amdgcn-amd-amdhsa--%s\"\n" \
            % (isaToGfx(self.kernel.isa))
        kStr += ".text\n"
        kStr += ".protected %s\n" % self.name
        kStr += ".globl %s\n" % self.name
        kStr += ".p2align 8\n"
        kStr += ".type %s,@function\n" % self.name
        kStr += ".section .rodata,#alloc\n"
        kStr += ".p2align 6\n"
        kStr += ".amdhsa_kernel %s\n" % self.name
        kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_segment_ptr 1\n"
        if self.accumOffset != None:
            kStr += kdIndent + ".amdhsa_accum_offset %u // accvgpr offset\n" % self.accumOffset
        kStr += kdIndent + ".amdhsa_next_free_vgpr %u // vgprs\n" % self.totalVgprs
        kStr += kdIndent + ".amdhsa_next_free_sgpr %u // sgprs\n" % self.totalSgprs
        kStr += kdIndent + ".amdhsa_group_segment_fixed_size %u // lds bytes\n" % self.groupSegSize
        if self.archCaps["HasWave32"]:
            if self.kernel.wavefrontSize == 32:
                kStr += kdIndent + ".amdhsa_wavefront_size32 1 // 32-thread wavefronts\n"
            else:
                kStr += kdIndent + ".amdhsa_wavefront_size32 0 // 64-thread wavefronts\n"
        kStr += kdIndent + ".amdhsa_private_segment_fixed_size 0\n"
        kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_x %u\n" % self.sgprWorkGroup[0]
        kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_y %u\n" % self.sgprWorkGroup[1]
        kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_z %u\n" % self.sgprWorkGroup[2]
        kStr += kdIndent + ".amdhsa_system_vgpr_workitem_id %u\n" % self.vgprWorkItem
        kStr += kdIndent + ".amdhsa_float_denorm_mode_32 3\n"
        kStr += kdIndent + ".amdhsa_float_denorm_mode_16_64 3\n"
        if self.enablePreloadKernArgs:
            numWorkgroupSgpr = self.sgprWorkGroup[0] + self.sgprWorkGroup[1] + self.sgprWorkGroup[2]
            kStr += kdIndent + ".amdhsa_user_sgpr_count %d\n" % (16-numWorkgroupSgpr)
            kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_preload_length %d\n" % (14-numWorkgroupSgpr)
            kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_preload_offset 0\n"
        kStr += ".end_amdhsa_kernel\n"
        kStr += ".text\n"
        kStr += block("Num VGPR   =%u"%self.originalTotalVgprs)
        kStr += block("Num AccVGPR=%u"%self.totalAgprs)
        kStr += block("Num SGPR   =%u"%self.totalSgprs)
        return kStr

    def prettyPrint(self, indent=""):
        ostream = ""
        ostream += "%s%s "%(indent, type(self).__name__)
        return ostream

class SignatureCodeMeta(Item):
    def __init__(self, name, kernArgsVersion, groupSegSize, flatWgSize, codeObjectVersion, totalVgprs = 0, totalSgprs=0):
        super().__init__(name)
        self.kernArgsVersion = kernArgsVersion
        self.groupSegSize = groupSegSize
        self.flatWgSize = flatWgSize
        self.codeObjectVersion = str(codeObjectVersion)
        self.totalVgprs = totalVgprs
        self.totalSgprs = totalSgprs
        self.offset = 0
        self.argList = []

    def setGprs(self, totalVgprs: int, totalSgprs: int):
        self.totalVgprs = totalVgprs
        self.totalSgprs = totalSgprs

    def __str__(self):
        kStr = ""
        kStr += ".amdgpu_metadata\n"
        kStr += "---\n"
        kStr += "custom.config:\n"
        kStr += "  InternalSupportParams:\n"
        kStr += "    KernArgsVersion: %d\n"%self.kernArgsVersion
        kStr += "amdhsa.version:\n"
        kStr += "  - 1\n"
        if self.codeObjectVersion == "4" or self.codeObjectVersion == "default":
            kStr += "  - 1\n"
        elif self.codeObjectVersion == "5":
            kStr += "  - 2\n"
        kStr += "amdhsa.kernels:\n"
        kStr += "  - .name: %s\n" % self.name
        kStr += "    .symbol: '%s.kd'\n" % self.name
        kStr += "    .language:                   OpenCL C\n"
        kStr += "    .language_version:\n"
        kStr += "      - 2\n"
        kStr += "      - 0\n"
        kStr += "    .args:\n"
        for i in self.argList:
            kStr += str(i)
        kStr += "    .group_segment_fixed_size:   %u\n" % self.groupSegSize
        kStr += "    .kernarg_segment_align:      %u\n" % 8
        kStr += "    .kernarg_segment_size:       %u\n" % (((self.offset+7)//8)*8) # round up to .kernarg_segment_align
        kStr += "    .max_flat_workgroup_size:    %u\n" % self.flatWgSize
        kStr += "    .private_segment_fixed_size: %u\n" % 0
        kStr += "    .sgpr_count:                 %u\n" % self.totalSgprs
        kStr += "    .sgpr_spill_count:           %u\n" % 0
        kStr += "    .vgpr_count:                 %u\n" % self.totalVgprs
        kStr += "    .vgpr_spill_count:           %u\n" % 0
        kStr += "    .wavefront_size:             %u\n" % self.kernel.wavefrontSize

        kStr += "...\n"
        kStr += ".end_amdgpu_metadata\n"
        kStr += "%s:\n" % self.name
        return kStr

    def addArg(self, name: str, kind: SignatureValueKind, type: str, addrSpaceQual: Optional[str]=None):
        sa = _SignatureArgument(self.offset, name, kind, type, addrSpaceQual)
        self.argList.append(sa)
        self.offset += sa.size

    def prettyPrint(self, indent=""):
        ostream = ""
        ostream += "%s%s "%(indent, type(self).__name__)
        return ostream

class SignatureBase(Item):
    def __init__(self, kernelName, kernArgsVersion, codeObjectVersion, groupSegmentSize, sgprWorkGroup, \
        vgprWorkItem, flatWorkGroupSize, totalVgprs: int=0, totalAgprs: int=0, \
        totalSgprs: int=0, preloadKernArgs: bool=False) -> None:
        super().__init__(kernelName)

        # Internal data
        self.kernelDescriptor = _SignatureKernelDescriptor(name=kernelName,
                                                                totalVgprs=totalVgprs,
                                                                totalAgprs=totalAgprs,
                                                                totalSgprs=totalSgprs,
                                                                groupSegSize=groupSegmentSize,
                                                                sgprWorkGroup=sgprWorkGroup,
                                                                vgprWorkItem=vgprWorkItem,
                                                                preloadKernArgs=preloadKernArgs)
        self.codeMeta = SignatureCodeMeta(name=kernelName,
                                                kernArgsVersion=kernArgsVersion,
                                                groupSegSize=groupSegmentSize,
                                                flatWgSize=flatWorkGroupSize,
                                                codeObjectVersion=codeObjectVersion,
                                                totalVgprs=totalVgprs,
                                                totalSgprs=totalSgprs)

        # Comment description
        self.descriptionTopic = None
        self.descriptionList = []

    def setGprs(self, totalVgprs: int, totalAgprs: int, totalSgprs: int):
        self.kernelDescriptor.setGprs(totalVgprs=totalVgprs, totalAgprs=totalAgprs, \
            totalSgprs=totalSgprs)
        self.codeMeta.setGprs(totalVgprs=totalVgprs, totalSgprs=totalSgprs)

    def addArg(self, name: str, kind: SignatureValueKind, type: str, addrSpaceQual: Optional[str]=None):
        self.codeMeta.addArg(name, kind, type, addrSpaceQual)

    def addDescriptionTopic(self, text: str):
        self.descriptionTopic = TextBlock(block3Line(text))

    def addDescriptionBlock(self, text: str):
        self.descriptionList.append(TextBlock(block(text)))

    def addDescription(self, text: str):
        self.descriptionList.append(TextBlock(slash(text)))

    def getNextFreeVgpr(self) -> int:
        return self.kernelDescriptor.getNextFreeVgpr()

    def getNextFreeSgpr(self) -> int:
        return self.kernelDescriptor.getNextFreeSgpr()

    def clearDescription(self, text: str):
        self.descriptionList = []

    def __str__(self):
        kStr = ""
        kStr += str(self.kernelDescriptor)
        if self.descriptionTopic != None:
            kStr += str(self.descriptionTopic)
        for i in self.descriptionList:
            kStr += str(i)
        kStr += str(self.codeMeta)
        return kStr

    def prettyPrint(self, indent=""):
        ostream = ""
        ostream += "%s%s "%(indent, type(self).__name__)
        return ostream

########################################
# Signatures
########################################

class KernelBody(Item):
    def __init__(self, name) -> None:
        super().__init__(name)

    def addSignature(self, signature: SignatureBase):
        self.signature = signature

    def addBody(self, body: Module):
        self.body = body

    def setGprs(self, totalVgprs: int, totalAgprs: int, totalSgprs: int):
        self.totalVgprs = totalVgprs
        self.totalAgprs = totalAgprs
        self.totalSgprs = totalSgprs
        self.signature.setGprs(totalVgprs=totalVgprs, totalAgprs=totalAgprs, \
            totalSgprs=totalSgprs)

    def getNextFreeVgpr(self) -> int:
        return self.signature.getNextFreeVgpr()

    def getNextFreeSgpr(self) -> int:
        return self.signature.getNextFreeSgpr()

    def __str__(self) -> str:
        kStr = str(TextBlock(block3Line("Begin Kernel")))
        kStr += str(self.signature)
        kStr += str(self.body)
        return kStr
