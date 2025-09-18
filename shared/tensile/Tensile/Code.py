################################################################################
#
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import print_function
from .Common import globalParameters, printExit
import ctypes
# Global to print module names around strings
printModuleNames = 0

class Item:
  """
  Base class for Modules, Instructions, etc
  Item is a atomic collection of or more instructions and commentsA
  """
  pass

  def toStr(self):
    return str(self)

  def countType(self,ttype):
    return int(isinstance(self, ttype))

  def prettyPrint(self, indent=""):
    ostream = ""
    ostream += "%s%s "%(indent, type(self).__name__)
    ostream += str(self)
    return ostream

class Module(Item):
  """
  Modules contain lists of text instructions, Inst objects, or additional modules
  They can be easily converted to string that represents all items in the list
  and can be mixed with standard text.
  The intent is to allow the kernel writer to express the structure of the
  code (ie which instructions are a related module) so the scheduler can later
  make intelligent and legal transformations.
  """
  def __init__(self, name=""):
    self.name     = name
    self.itemList = []
    self.tempVgpr = None

  def findNamedItem(self, targetName):
    return next((item for item in self.itemList if item.name==targetName), None)

  def __str__(self):
    s = ""
    if printModuleNames:
      s += "// %s { \n" % self.name
    s += "".join([str(x) for x in self.itemList])
    if printModuleNames:
      s += "// } %s\n" % self.name
    return s

  def addCode(self, item):
    """
    Add specified item to the list of items in the module.
    Item MUST be a Item (not a string) - can use
    addText(...)) to add a string.
    All additions to itemList should use this function.

    Returns item to facilitate one-line create/add patterns
    """
    #assert (isinstance(item, Item)) # for debug
    if isinstance(item,Item):
      self.itemList.append(item)
    elif isinstance(item,str):
      self.addCode(TextBlock(item))
    else:
      assert 0, "unknown item type (%s) for Module.addCode. item=%s"%(type(item), item)
    return item

  def addComment0(self, comment):
    """
    Convenience function to format arg as a comment and add TextBlock item
    This comment is a single line /* MYCOMMENT  */
    """
    self.addCode(TextBlock("/* %s */\n"%comment))

  def addComment1(self, comment):
    """
    Convenience function to format arg as a comment and add TextBlock item
    This comment is a blank line followed by /* MYCOMMENT  */
    """
    self.addCode(TextBlock("\n/* %s */\n"%comment))

  def addInst(self, *args):
    """
    Convenience function to construct a single Inst and add to items
    """
    self.addCode(Inst(*args))

  def addText(self,text):
    """
    Convenience function to construct a TextBlock and add to items
    """
    self.addCode(TextBlock(text))

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
      mod2.addCode(Code.Inst("bogusInst", "comments"))
      mod3 = Code.Module("Module-lvl3")
      mod3.addCode(Code.TextBlock("bogusTextBlock\nbogusTextBlock2\nbogusTextBlock3"))
      mod3.addCode(Code.GlobalReadInst("bogusGlobalReadInst", "comments"))
      mod2.addCode(mod3)
      mod1.addCode(mod2)
      mod1.addCode(Code.Inst("bogusInst", "comments"))
      mod1.addCode(mod2)

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

  def countType(self,ttype):
    """
    Count number of items with specified type in this Module
    Will recursively count occurrences in submodules
    (Overrides Item.countType)
    """
    count=0
    for i in self.itemList:
      if isinstance(i, Module):
        count += i.countType(ttype)
      else:
        count += int(isinstance(i, ttype))
    return count

  def count(self):
    count=0
    for i in self.itemList:
      if isinstance(i, Module):
        count += i.count()
      else:
        count += 1
    return count

  def items(self):
    """
    Return list of items in the Module
    Items may be other Modules, TexBlock, or Inst
    """
    return self.itemList

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
  def __init__(self, name=None):
    Module.__init__(self,name)
    self.header = Module("header")
    self.middle = Module("middle")
    self.footer =  Module("footer")

    self.addCode(self.header)
    self.addCode(self.middle)
    self.addCode(self.footer)


class Label (Item):
  """
  Label that can be the target of a jump.
  """
  def __init__(self, labelNum, comment):
    self.labelNum = labelNum
    self.comment = comment

  def __str__(self):
    t = "label_%04u:" % (self.labelNum)
    if self.comment:
      t += "  /// %s" % self.comment
    t += "\n"
    return t


class TextBlock(Item):
  """
  An unstructured block of text that can contain comments and instructions
  """
  def __init__(self,text):
    assert(isinstance(text, str))
    self.text = text
    self.name = text

  def __str__(self):
    return self.text

  def prettyPrint(self, indent=""):
    ostream = ""
    ostream += "%s%s "%(indent, type(self).__name__)
    l = [_i for _i in str(self).split("\n") if len(_i)>0]
    l.insert(0, "")
    ostream += "%s"%(("\n"+indent.replace("|-", "| |-")).join(l))
    ostream += "\n"
    return ostream

class Inst(Item):
  """
  Inst is a single instruction and is base class for other instructions.
  Currently just stores text+comment but over time may grow
  """
  def __init__(self, *args):
    params = args[0:len(args)-1]
    comment = args[len(args)-1]
    assert(isinstance(comment, str))
    formatting = "%s"
    if len(params) > 1:
      formatting += " %s"
    for i in range(0, len(params)-2):
      formatting += ", %s"
    instStr = formatting % (params)
    self.text = self.formatWithComment(instStr, comment)

  def formatWithComment(self, instStr, comment):
    return "%-50s // %s\n" % (instStr, comment)

  def __str__(self):
    return self.text
class WaitCnt (Module):
  """
  Construct a waitcnt from specified lgkmcnt and vmcnt:
  lgkmcnt, vmcnt:
    if -1 then will not be added to the wait term.

  If lgkmcnt=vmcnt= -1 then the waitcnt is a nop and
  an instruction with a comment is returned.
  """
  def __init__(self, version,lgkmcnt=-1,vmcnt=-1,comment=""):
    super().__init__("wait")

    self.version = version
    self.lgkmcnt = lgkmcnt
    self.vmcnt   = vmcnt
    self.comment = "lgkmcnt={} vmcnt={}".format(lgkmcnt, vmcnt) + comment

    # let this derived class play nicely with Module.prettyPrint()
    self.__dict__.update(self.instructions().__dict__)

  def instructions(self):
    rv = Module()
    main_args = []
    wait_store = False
    if self.lgkmcnt != -1:
      currentIsa    = globalParameters["CurrentISA"]
      maxLgkmcnt    = globalParameters["AsmCaps"][currentIsa]["MaxLgkmcnt"]
      seperateVscnt = globalParameters["ArchCaps"][currentIsa]["SeparateVscnt"]
      wait_store    = True
      main_args += ["lgkmcnt(%u)" % (min(self.lgkmcnt,maxLgkmcnt))]

    if self.vmcnt != -1:
      main_args += ["vmcnt(%u)" % self.vmcnt]

    if len(main_args) > 0:
      rv.addInst("s_waitcnt", *main_args, self.comment)
      if wait_store and seperateVscnt and self.vmcnt != -1:
        rv.addInst("s_waitcnt_vscnt", "null", self.vmcnt, "writes")
    else:
      rv.addComment0(self.comment)

    return rv

  def __str__(self):
    return str(self.instructions())

# uniq type that can be used in Module.countType
class GlobalReadInst (Inst):
  def __init__(self,*args):
    Inst.__init__(self,*args)

# uniq type that can be used in Module.countType
class LocalWriteInst (Inst):
  def __init__(self,issuelatency,*args):
    self.IssueLatency = issuelatency
    Inst.__init__(self,*args)

# uniq type that can be used in Module.countType
class LocalReadInst (Inst):
  def __init__(self,issuelatency,readToTempVgpr,*args):
    self.IssueLatency = issuelatency
    self.readToTempVgpr = readToTempVgpr
    Inst.__init__(self,*args)

################################################################################
# MFMA Instruction
################################################################################
class  MFMAInst (Inst):

  """
  Construct a MFMA instruction from specified precision Type, aIndex, bIndex, PLR, innerUnroll:

  dataType:
  aIndex:  index value from range (0, kernel["ThreadTile0"])
  bIndex:  index value from range (0, kernel["ThreadTile1"])

  PLR:     valida values 0,1

  usage Module.addCode(Code.MFMAInst())

  """
  def  __init__(self,kernel,aIdx,bIdx,PLRval,innerUnroll):
       self.endLine = ""
       self.version = globalParameters["CurrentISA"]
       self.kernel  = kernel
       self.aIdx    = aIdx
       self.bIdx    = bIdx
       self.PLR     = PLRval
       self.innerUnroll = innerUnroll

  def __str__(self):
      # single precision
      kStr = ""
      numOfRowsperMfma = 1
      numOfRowInsts = self.kernel["ThreadTile0"]/numOfRowsperMfma
      #numOfColInsts = kernel["ThreadTile1"]/kernel["MatrixInstN"]
      numOfDstRgs = (self.kernel["MatrixInstN"] * self.kernel["MatrixInstM"] * self.kernel["MatrixInstB"] // self.kernel["WavefrontSize"])
      if self.kernel["ProblemType"]["DataType"].isSingle():
        for iui in range(0, self.innerUnroll):
           cStr = "a[(%u+%u*%u)*%u):((((%u+%u*%u)*%u)+%u)-1)]" % (self.aIdx,self.bIdx,numOfRowInsts,numOfDstRgs,self.aIdx,numOfDstRgs,self.bIdx,numOfRowInsts,numOfDstRgs)
           aStr = "v[%s+%u]" \
               % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
           bStr = "v[%s+%u]" \
               % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
           kStr += "v_mfma_f32_%ux%ux%uf32 %s, %s, %s, %s%s" % (self.kernel["MatrixInstM"], self.kernel["MatrixInstN"], self.kernel["MatrixInstK"], cStr, aStr, bStr, cStr, self.endLine)
      else:
        printExit("Assembly doesn't support %s" % self.kernel["ProblemType"]["DataType"])

      return self.formatWithComment(kStr, "")

  def getLatency(self):
      # return latency in cycles
      return  (self.kernel["MatrixInstM"] // 4 ) * 8


################################################################################
# Mac Instruction
# can be generic as VALU instruction
# implement later generic
################################################################################
class  MacInst (Inst):
  """
  Construct a mac instruction from specified dataType, aIndex, bIndex, PLR, innerUnroll:

   - dataType:
   - aIndex:  index value from range (0, kernel["ThreadTile0"])
   - bIndex:  index value from range (0, kernel["ThreadTile1"])

  PLR:     valida values 0,1

  usage `Module.addCode(Code.MacInst())`
  """

  def  __init__(self,kernel,aIdx,bIdx,PLRval,innerUnroll):
       self.endLine = ""
       self.version = globalParameters["CurrentISA"]
       self.kernel  = kernel
       self.aIdx    = aIdx
       self.bIdx    = bIdx
       self.PLR     = PLRval
       self.innerUnroll = innerUnroll

  #def toCodeInst(self,kernel,aIdx,bIdx,PLRval,innerUnroll):
  def __str__(self):
      # half precision
      kStr = ""
      if self.kernel["ProblemType"]["DataType"].isHalf():
        if self.version == (8,0,3):
          for b in range(self.bIdx*2, (self.bIdx+1)*2):
            for a in range(self.aIdx*2, (self.aIdx+1)*2):
              for iui in range(0, self.innerUnroll):
                    # v_mac_f16 or v_fma_f16
                    cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"])
                    aStr = "v[%s+%u]" \
                        % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
                    bStr = "v[%s+%u]" \
                        % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
                    kStr += "v_mac_f16 %s, %s, %s\n" % (cStr, aStr, bStr) # FIXME op_sel
        elif self.version == (9,0,0):
          if self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
            # we treat HighPrecisionAccumulate as expanded packed math
            #b = self.bIdx*2
            #a = self.aIdx*2
            if self.kernel["LocalDotLayout"] > 1 and self.innerUnroll == 2:    # Only supports LocalDotLayout == 2 for now
              cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 0
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I0"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I0"%self.PLR, self.bIdx)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I1"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I0"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 0
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I0"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I1"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 1
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I1"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I1"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              #kStr += self.bomb(-13)
              """
              ignore this, not quite correct for mixed precision
              D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
              D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
              C[0] = A[0]*B[0]+D[0]
              C[1] = A[1]*B[1]+D[1]
              """
            else:
              for iui in range(0, self.innerUnroll):
                 cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
                 cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 0
                 aStr = "v[%s+%u]" \
                     % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
                 bStr = "v[%s+%u]" \
                     % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
                 kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iui=%u%s" % (cStr, aStr, bStr, cStr, cidx, iui, self.endLine)
                 cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
                 cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
                 kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                 cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 0
                 cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
                 kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                 cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 1
                 cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
                 kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                 """
                    ignore this, not quite correct for mixed precision
                    D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                    D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                    C[0] = A[0]*B[0]+D[0]
                    C[1] = A[1]*B[1]+D[1]
                 """
          else:
            #b = self.bIdx*2
            #a = self.aIdx*2
            for iui in range(0, self.innerUnroll):
              cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # /2 b/c of 2 f16's per 32-bit vgpr
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
              kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,0,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
              cStr = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
              """
               D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
               D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
               C[0] = A[0]*B[0]+D[0]
               C[1] = A[1]*B[1]+D[1]
              """
        elif self.version == (9,0,6):
          if self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
             # we treat HighPrecisionAccumulate as expanded packed math
            #b = self.bIdx*2
            #a = self.aIdx*2
            if self.kernel["LocalDotLayout"] > 1 and self.innerUnroll == 2:    # Only supports LocalDotLayout == 2 for now
              cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 0
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I0"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I0"%self.PLR, self.bIdx)
              kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I1"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I0"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
              kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 0
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I0"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I1"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 1
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I1"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I1"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              #kStr += self.bomb(-13)
              """
              ignore this, not quite correct for mixed precision
              D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
              D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
              C[0] = A[0]*B[0]+D[0]
              C[1] = A[1]*B[1]+D[1]

              """
            else:
              for iui in range(0, self.innerUnroll):
                cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 0
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iui=%u%s" % (cStr, aStr, bStr, cStr, cidx, iui, self.endLine)
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iui=%u%s" % (cStr, aStr, bStr, cStr, cidx, iui, self.endLine)
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
                cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[1,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 0
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 1
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                """
                ignore this, not quite correct for mixed precision
                D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                C[0] = A[0]*B[0]+D[0]
                C[1] = A[1]*B[1]+D[1]
                """
                  #kStr += self.bomb(-13)
          else:
            #b = self.bIdx*2
            #a = self.aIdx*2
            for iui in range(0, self.innerUnroll):
              cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # /2 b/c of 2 f16's per 32-bit vgpr
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
              kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,0,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)

              cStr = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
              """
              D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
              D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
              C[0] = A[0]*B[0]+D[0]
              C[1] = A[1]*B[1]+D[1]
              """
        else:
          raise NotImplementedError("Half-precision not supported for arch=%u" % self.version )

      # integer i8x4
      elif self.kernel["ProblemType"]["DataType"].isInt8x4():
        if self.version == (8,0,3):
          kStr += "// int8 not implemented yet for gfx803:"
        elif self.version == (9,0,0):
          kStr += "// int8 not implemented yet for gfx900:"
        elif self.version == (9,0,6):
          for iui in range(0, self.innerUnroll):
            cidx = self.aIdx + self.bIdx*self.kernel["ThreadTile0"] + 0
            cStr = "v[%s+%u+%u*%u]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"])
            aStr = "v[%s+%u]"       % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
            bStr = "v[%s+%u]"       % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
            kStr += "v_dot4_i32_i8  %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
      # single precision
      elif self.kernel["ProblemType"]["DataType"].isSingle():
        for iui in range(0, self.innerUnroll):
          cStr = "v[%s+%u+%u*%u]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"])
          aStr = "v[%s+%u]" \
              % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
          bStr = "v[%s+%u]" \
              % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
          #if a==0 and b==0:
          #  kStr += dump(aStr)
          kStr += "_v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)
          ##if macIdx == self.kernel["PerformanceWaitLocation"]:
          ##    kStr += "s_waitcnt lgkmcnt(%u) // extra wait for performance%s" \
          ##        % (self.kernel["PerformanceWaitCount"], self.endLine)
          ##if macIdx == self.kernel["PerformanceSyncLocation"]:
          ##    kStr += "s_barrier // extra barrier for performance%s" \
          ##        % (self.endLine)
          ##macIdx += 1

      # double precision
      elif self.kernel["ProblemType"]["DataType"].isDouble():
        for iui in range(0, self.innerUnroll):
           cStr = "v[%s+(%u+%u*%u)*2:(%s+%u+%u*%u)*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], "vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"])
           aStr = "v[%s+%u*2:%s+%u*2+1]" \
               % ("vgprValuA_X%u_I%u"%(self.PLR,iui) , self.aIdx, "vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
           bStr = "v[%s+%u*2:%s+%u*2+1]" \
               % ("vgprValuB_X%u_I%u"%(self.PLR,iui) , self.bIdx, "vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
           kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)

      # other precision
      else:
        printExit("Assembly doesn't support %s" % self.kernel["ProblemType"]["DataType"])

      return self.formatWithComment(kStr, "")

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

  def desc(self):
    return "hex: {}\n".format(self) + self.fields.desc()

class SrdUpperFields9XX(BitfieldStructure):
  _fields_ = [("dst_sel_x",      ctypes.c_uint, 3),
              ("dst_sel_y",      ctypes.c_uint, 3),
              ("dst_sel_z",      ctypes.c_uint, 3),
              ("dst_sel_w",      ctypes.c_uint, 3),
              ("num_format",     ctypes.c_uint, 3),
              ("data_format",    ctypes.c_uint, 4),
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


class SrdUpperFields11XX(BitfieldStructure):
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


class SrdUpperValue11XX(BitfieldUnion):
  _fields_ = [("fields", SrdUpperFields11XX), ("value", ctypes.c_uint32)]

  @classmethod
  def default(cls):
    return cls(fields=SrdUpperFields11XX.default())


def SrdUpperValue(isa):
  if isa[0] == 11:
    return SrdUpperValue11XX.default()
  elif isa[0] == 10:
    return SrdUpperValue10XX.default()
  else:
    return SrdUpperValue9XX.default()



class OpTemplate(Module):
    """
    Base template for high level operator (mem/alu), template contain data and code
    section for implementing high level operator;
    resources required for high level operator are already accounted in main parts
    kernel section

    Usage: high level operator like unary,binary operators opearting on tensor dimension or
           memory operator that stores/loads tensor from near/far memory

    Initially monolithic code segments doing high level operator  vpgr/sgpr (temp ones ) are placeholders
    must be replaced when its actually placed in main kernel


    """

    ## constructor
    def __init__(self, name=""):
      self.name     = name
      ## list order of operations
      ## Module key "LaddrCalcA", "lrda", "lwra", "StoreC", "loadC",
      ## order of code is important
      self.itemList = []   ## list of Code Modules
      self.tmpSgpr  = None
      self.tmpVgpr  = None

    def __str__(self):
      s = ""
      if printModuleNames:
        s += "// %s { \n" % self.name
      s += "".join([str(x) for x in self.itemList])
      if printModuleNames:
        s += "// } %s\n" % self.name
      return s

    def findNamedCode(self, targetName):
      return next((Moditem for Moditem in self.itemList if Moditem.name==targetName), None)

    def addModule(self, ModItem):
      """
      Add specified Code modules to the list of Modules in the opTemplate
      ModItem MUST be a Module, list of instructions not string
      returns Module to facilitate one-line create/add patterns
      """
      if isinstance(ModItem,Module):
        #self.itemList.append(ModItem)
        self.itemList.extend(ModItem.itemList)
      else:
        assert 0, "unknown ModItem type (%s) for OpTemplate.addCode. Moditem=%s"%(type(ModItem), ModItem)
      return ModItem

      def addTempSgpr(self, sgpr):
        self.tmpSgpr = sgpr

      def addTempVgpr(self, vgpr):
        self.tmpVgpr = vgpr

class MemOpTemplate(OpTemplate):
    """
    template for local/global data movement code sections
    list should have sequence of code modules supporting data movement,
    including offset calculation , offset increment, load/store

    current code sections are expected to follow program consistency (no out of order in scheduling them)

    This needs further refinement- handling temp registers in code
    temporary register(s) used at the time of code generation(S) need to be replaced when its called
    temp register in instruction should use _sgpr%len_  or _vgpr%len_  len determines number of temp register

    before using code , allocate number of registers required for code

    """
    def __init__(self, name=""):
      self.name = name
      self.itemList = []   ## list of  COde Modules
      self.tmpSgpr  = None
      self.tmpVgpr  = None
