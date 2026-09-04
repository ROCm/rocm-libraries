################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

from copy import deepcopy

from .Activation import ActivationType
from .KernelWriterBase import KernelWriterBase

class KernelWriterActivationEnumHeader(KernelWriterBase):

  def __init__(self, state):
    super().__init__()
    self.state["ProblemType"] = deepcopy(state["ProblemType"])

    self.actGradientPrefix = ""
    self.actExportType = ActivationType.Export.NORMAL
    if self.state["ProblemType"]["Gradient"]:
      self.actGradientPrefix = "Gradient"
      self.actExportType = ActivationType.Export.GRADONLY

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()

  def keys(self):
    return self.getKernelName()

  @staticmethod
  def kernelName(solution):
    state = solution._state if hasattr(solution, "_state") else solution.state
    s = "Gradient" if state["ProblemType"]["Gradient"] else ""
    return "Tensile%sActivationEnum_%s"%(s,
                                         state["ProblemType"]["ActivationComputeDataType"].toChar())

  def getKernelName(self):
    return KernelWriterActivationEnumHeader.kernelName(self)



  def getSourceFileString(self):
    fileString = "// This is a dummy file."
    return (0, fileString)

  def getHeaderFileString(self):
    fileString = "" # CHeader
    activationCDataType = self.state["ProblemType"]["ActivationComputeDataType"]
    supportedBy = ActivationType.SupportedBy.ALL if self.state["ProblemType"]["ActivationType"] == 'all' else ActivationType.SupportedBy.HIPBLASLT
    enumName = "%sActivationType_%s"%(self.actGradientPrefix, activationCDataType.toChar())
    fileString += "namespace Tensile {\n"
    fileString += "enum class %s : uint32_t\n"%enumName
    fileString += "{\n"
    enumList = ActivationType.getEnumStrList(activationCDataType, supportedBy, exportType=self.actExportType)
    for idx, enumStr in enumerate(enumList):
      fileString += "  %s = %s,\n"%(ActivationType(enumStr).toEnum(), ActivationType.getEnumIndex(enumStr))
    fileString += "};\n"
    fileString += "}  // End of namespace Tensile\n"

    return fileString
