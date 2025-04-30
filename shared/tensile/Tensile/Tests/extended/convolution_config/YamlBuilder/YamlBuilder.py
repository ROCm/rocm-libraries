################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import subprocess
import copy, operator, pytest
from functools import reduce
from Tensile.SolutionStructs import ConvProblem
from Tensile.Utilities.ConditionalImports import yamlDumper

import yaml


class Solutions:

    # If adding new solutions, update 'solutions' in conftest.py

    @classmethod
    def commonSetup(cls):
        return {
            "InitialSolutionParameters": None,
            "BenchmarkCommonParameters": [{"EdgeType": ["ShiftPtr"]}],
            "ForkParameters": None,
            "BenchmarkForkParameters": None,
            "JoinParameters": None,
            "BenchmarkJoinParameters": None,
            "BenchmarkFinalParameters": None
        }

    @classmethod
    def src1(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [0]},
                    {"KernelLanguage": ["Source"]},
                    {"ThreadTile": [
                        [ 2, 2 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1 ]
                        ]},
                    {"DepthU": [8]},
                    {"PackBatchDims": [1]}, # required to handle some Backward-Weights cases
                    {"PackSummationDims": [1]}, # required to handle padding cases
                    {"GlobalReadVectorWidth": [1]},
                    {"VectorWidth": [1]},
                ]

        return s

    @classmethod
    def src5_gsu(cls):
        # Has GSU configs + some other options.
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [0]},
                    {"KernelLanguage": ["Source"]},
                    {"ThreadTile": [
                        [ 8, 8 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 16, 1 ]
                        ]},
                    {"DepthU": [4]},
                    {"PackBatchDims": [0,1]},
                    {"PackSummationDims": [0,1]},
                    {"GlobalSplitU": [1,2,4,7,17]},
                    {"GlobalReadVectorWidth": [-1]},
                    {"VectorWidth": [1,-1]},
                ]

        return s

    @classmethod
    def asm1(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [0]},
                    {"KernelLanguage": ["Assembly"]},
                    {"ThreadTile": [
                        [ 4, 4 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1 ]
                        ]},
                    {"DepthU": [8]},
                    {"PackBatchDims": [1]}, # required to handle some Backward-Weights cases
                    {"PackSummationDims": [1]}, # required to handle padding cases
                    {"GlobalReadVectorWidth": [-1]},
                    {"VectorWidth": [1]},
                    {"FractionalLoad": [1]}
                ]

        return s

    @classmethod
    def asm3_pbd(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [1]},
                    {"KernelLanguage": ["Assembly"]},
                    {"ThreadTile": [
                        [ 8, 4 ],
                        [ 4, 8 ],
                        [ 8, 8 ]
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1],
                        [ 16, 8, 1]
                        ]},
                    {"DepthU": [8]},
                    {"GlobalReadVectorWidth": [-1]},
                    {"PackSummationDims": [0,1]},
                    {"VectorWidth": [1,4]},
                    {"FractionalLoad": [1]},
                    {"PackBatchDims": [0,1]},
                ]

        return s

    @classmethod
    def asm3_splitu(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [1]},
                    {"KernelLanguage": ["Assembly"]},
                    {"ThreadTile": [
                        [ 4, 4 ],
                        [ 4, 8 ],
                        [ 8, 4 ],
                        [ 8, 8],
                        ]},
                    {"WorkGroup": [
                        [  8, 8, 1],
                        #[  8, 8, 2],
                        #[  8, 8, 4],
                        #[ 16, 8, 2],
                        #[ 16, 16, 2],
                        ]},
                    {"DepthU": [8]},
                    {"PackSummationDims": [0,1]},
                    {"GlobalReadVectorWidth": [-1]},
                    {"GlobalSplitU": [1,2,3,4,8,17]},
                    {"VectorWidth": [1,4]},
                    {"PackBatchDims": [1]},
                    {"FractionalLoad": [1]}
                ]

        return s

    @classmethod
    def asm3_mi(cls):
        s = cls.commonSetup()

        s["ForkParameters"] = \
                [
                    {"PrefetchGlobalRead": [1]},
                    {"KernelLanguage": ["Assembly"]},
                    {"MatrixInstruction": [
                        [32,32,1,2]
                        ]},
                    {"ThreadTile": [
                        [ 1, 32 ]
                        ]},
                    {"WorkGroup": [
                        [ 16, 16, 1],
                        [ 64, 4, 1]
                        ]},
                    {"DepthU": [8]},
                    {"GlobalReadVectorWidth": [-1,2]},
                    {"PackSummationDims": [0,1]},
                    {"VectorWidth": [1,4]},
                    {"FractionalLoad": [1]},
                    {"PackBatchDims": [0,1]},
                ]

        return s

    @classmethod
    def defaultSolution(cls):
        return cls.asm3_pbd

class YamlBuilder:

    def __init__(self, doc):
        self.doc = doc

    def write(self, fname):
        with open(str(fname), "w") as f:
            yaml.dump(self.doc, f, yamlDumper, default_flow_style=None)

    @classmethod
    def findAvailableArchs(self):
        availableArchs = []
        rocmpath = "/opt/rocm"
        if "ROCM_PATH" in os.environ:
            rocmpath = os.environ.get("ROCM_PATH")
        if "TENSILE_ROCM_PATH" in os.environ:
            rocmpath = os.environ.get("TENSILE_ROCM_PATH")
        rocmAgentEnum = os.path.join(rocmpath, "bin/rocm_agent_enumerator")
        output = subprocess.check_output([rocmAgentEnum, "-t", "GPU"])
        lines = output.decode().splitlines()
        for line in lines:
            line = line.strip()
            if not line in availableArchs:
                availableArchs.append(line)
        return availableArchs

    @classmethod
    def Header(cls,debug):
        rv= \
        {
            "GlobalParameters":
            {
                "MinimumRequiredVersion": "4.2.0",
                "ForceRedoBenchmarkProblems": True,
                "ForceRedoLibraryLogic": True,
                "ForceRedoLibraryClient": True,
                "CMakeBuildType": "RelWithDebInfo",
                "EnqueuesPerSync": 1,
                "SyncsPerBenchmark": 1,
                "LibraryPrintDebug": False,
                "NumElementsToValidate": 1000,
                "ValidationMaxToPrint": 4,
                "ValidationPrintValids": False,
                "ShortNames": False,
                "MergeFiles": True,
                "KernelTime": True,
                "SolutionSelectionAlg": 1,
                "DataInitTypeAlpha": 1, # use optimized OptNoLoadLoop, if available
                "DataInitTypeC": 4, # NANs
                "DataInitTypeD": 4, # NANs
                #"CpuThreads": 0,
            }
        }

        if debug:
          rv["GlobalParameters"]["CMakeBuildType"] = "Debug"
          rv["GlobalParameters"]["CpuThreads"] = 0

        return rv

    @classmethod
    def makeValidProblem(cls, conv, problem, strides=(2,3,4),  pads=(-1,-1,-1)):
        if conv.cc.stride[0] == -1:
            problem['v'] = strides[0]
        if conv.cc.stride[1] == -1:
            problem['u'] = strides[1]

        if conv.formatNumSpatialDims==3:
            if conv.cc.stride[2] == -1:
                problem['#'] = strides[2]

        padChars='qp$'
        for i in range(conv.formatNumSpatialDims):
            if conv.cc.padStart[i] == -1:
                problem[padChars[i]] = conv.cc.fil[i]-1 if pads[i] == -1 else pads[i]
            if conv.cc.padEnd[i] == -1:
                problem[padChars[i]+'_'] = conv.cc.fil[i]-1 if pads[i] == -1 else pads[i]

        return problem

    @classmethod
    def genSpatials(cls, conv, spatialRange):
        spatials = []
        for h in spatialRange:
            for w in spatialRange:
                if conv.formatNumSpatialDims==2:
                    spatials.append((h,w))
                elif conv.formatNumSpatialDims==3:
                    for d in [7]:
                        spatials.append((h,w,d))
                else:
                    raise RuntimeError('unknown formatNumSpatialDims=%d'%conv.formatNumSpatialDims)
        return spatials

    @classmethod
    def genProblems(cls, conv, nRange, ckRange, spatialRange):
        problems = []
        spatials = cls.genSpatials(conv, spatialRange)
        for n in nRange:
            for c in ckRange:
                for k in ckRange:
                    for s in spatials:
                        problem = {'n':n, 'c':c, 'h':s[1], 'w':s[0], 'k':k}
                        problem = cls.makeValidProblem(conv, problem)
                        if conv.formatNumSpatialDims==3:
                            problem['d'] = s[2]
                        problems.append(problem)
        return problems

    @staticmethod
    def memSize(indexAssignments, problem):
        """
        Return max memory required for specified index assignments in list of exacts
        """
        maxSize=0
        for exact in problem.sizes: # should account for strides
            size = reduce(operator.mul,[exact[idx] for idx in indexAssignments], 1)
            if size > maxSize:
                maxSize = size

        return maxSize

    @classmethod
    def ProblemSizesResNet(cls, conv, problemType, problemLevel):
        problems = []
        n=12
        for (c,h,w,k, x,y, p,q ,u,v) in (
                [1024,14,14,2048,1,1,0,0,0,0],
                [1024,14,14,256,1,1,0,0,0,0],
                [1024,14,14,512, 1,1, 0,0, 0,0],
                [128,28,28,128, 3,3, 1,1, 0,0],
                [128,28,28,512, 1,1, 0,0, 0,0],
                [2048,7,7,512, 1,1, 0,0, 0,0],
                [256,14,14,1024, 1,1, 0,0, 0,0],
                [256,14,14,256, 3,3, 1,1, 0,0],
                [256,56,56,128, 1,1, 0,0, 0,0],
                [256,56,56,512, 1,1, 0,0, 0,0],
                [256,56,56,64, 1,1, 0,0, 0,0],
                [3,230,230,64, 7,7, 0,0, 2,2],
                [512,28,28,1024, 1,1, 0,0, 0,0],
                [512,28,28,128, 1,1, 0,0, 0,0],
                [512,28,28,256, 1,1, 0,0, 0,0],
                [512,7,7,2048, 1,1, 0,0, 0,0],
                [512,7,7,512, 3,3, 1,1, 0,0],
                [64,56,56,256, 1,1, 0,0, 0,0],
                [64,56,56,64, 1,1, 0,0, 0,0],
                [64,56,56,64, 3,3, 1,1, 0,0],
             ):
            problem = {'n':n, 'c':c, 'h':h, 'w':w, 'k':k}
            problem = cls.makeValidProblem(conv, problem, strides=(u,v), pads=(p,q))
            problems.append(problem)
        return problems


    @classmethod
    def ProblemSizesInception(cls, conv, problemType, problemLevel):
        problems = []
        n=8

        for (count, c,h,w,k,x,y, p,q, u,v) in (
            (2,128,17,17,128,1,7,0,3,1,1),
            (2,128,17,17,128,7,1,3,0,1,1),
            (1,128,17,17,192,1,7,0,3,1,1),
            (1,128,17,17,192,7,1,3,0,1,1),
            (1,1280,8,8,192,1,1,0,0,1,1),
            (1,1280,8,8,320,1,1,0,0,1,1),
            (1,1280,8,8,384,1,1,0,0,1,1),
            (1,1280,8,8,448,1,1,0,0,1,1),
            (4,160,17,17,160,1,7,0,3,1,1),
            (4,160,17,17,160,7,1,3,0,1,1),
            (2,160,17,17,192,1,7,0,3,1,1),
            (2,160,17,17,192,7,1,3,0,1,1),
            (4,192,17,17,192,1,7,0,3,1,1),
            (1,192,17,17,192,3,3,0,0,2,2),
            (4,192,17,17,192,7,1,3,0,1,1),
            (1,192,17,17,320,3,3,0,0,2,2),
            (1,192,35,35,32,1,1,0,0,1,1),
            (1,192,35,35,48,1,1,0,0,1,1),
            (2,192,35,35,64,1,1,0,0,1,1),
            (1,2048,8,8,192,1,1,0,0,1,1),
            (1,2048,8,8,320,1,1,0,0,1,1),
            (1,2048,8,8,384,1,1,0,0,1,1),
            (1,2048,8,8,448,1,1,0,0,1,1),
            (1,256,35,35,48,1,1,0,0,1,1),
            (3,256,35,35,64,1,1,0,0,1,1),
            (1,288,35,35,384,3,3,0,0,2,2),
            (1,288,35,35,48,1,1,0,0,1,1),
            (4,288,35,35,64,1,1,0,0,1,1),
            (1,3,299,299,32,3,3,0,0,2,2),
            (1,32,147,147,64,3,3,1,1,1,1),
            (1,32,149,149,32,3,3,0,0,1,1),
            (4,384,8,8,384,1,3,0,1,1,1),
            (4,384,8,8,384,3,1,1,0,1,1),
            (2,448,8,8,384,3,3,1,1,1,1),
            (3,48,35,35,64,5,5,2,2,1,1),
            (4,64,35,35,96,3,3,1,1,1,1),
            (1,64,73,73,80,1,1,0,0,1,1),
            (2,768,17,17,128,1,1,0,0,1,1),
            (4,768,17,17,160,1,1,0,0,1,1),
            (1,80,73,73,192,3,3,0,0,1,1),
            (1,96,35,35,96,3,3,0,0,2,2),
            (3,96,35,35,96,3,3,1,1,1,1),
             ):
            #problem = {'n':n, 'c':c, 'h':h, 'w':w, 'k':k, 'x':x, 'y':y} # need to prune mismatches
            problem = {'count': count, 'n':n, 'c':c, 'h':h, 'w':w, 'k':k}
            problem = cls.makeValidProblem(conv, problem, strides=(u,v), pads=(p,q))
            problems.append(problem)
        return problems


    @classmethod
    def ProblemSizes(cls, conv, problemType, problemLevel):
        if conv.cc.spatial:
            spatialIn = conv.cc.spatial
        else:
            spatialIn = [14]*conv.formatNumSpatialDims

        if -1 in spatialIn:
            raise RuntimeError('Spatial must be completely specified, not "%s"'%spatialIn)
        if -1 in conv.cc.fil:
            raise RuntimeError('Filter must be completely specified, not "%s"'%conv.config['Filter'])

        problems = []
        problem = cls.makeValidProblem(conv,
                            {'n':8, 'c':32, 'h':spatialIn[1], 'w':spatialIn[0], 'k':16})
        if len(spatialIn)==3:
            problem['d'] = spatialIn[2]
        problems.append(problem)

        if not conv.cc.spatial:
          if problemLevel==2:
              problems += cls.genProblems(conv, nRange=(1,2,8), ckRange=[64], spatialRange=(7,14,56))
          elif problemLevel==3:
              problems += cls.genProblems(conv, nRange=(1,2,8), ckRange=range(127,129), spatialRange=(7,14,56))
          elif problemLevel==4:
              problems += cls.genProblems(conv, nRange=(1,2,8), ckRange=range(127,129), spatialRange=(7,56,73,111,194))

        #try:
        #    asize = cls.memSize(problemType["IndexAssignmentsA"], problems)
        #    bsize = cls.memSize(problemType["IndexAssignmentsB"], problems)
        #    dsize = cls.memSize(range(0,problemType["NumIndicesC"]), problems)
        #    print ("generated %d exact sizes.  ElementSizes: A=%d B=%d D=%d Total=%d" % \
        #            (len(problem), asize, bsize, dsize, asize+bsize+dsize))
        #except KeyError:
        #    None

        return problems

    @classmethod
    def ConvolutionVsContraction(cls, conv, solution, dataType):
        """
        Generates a YamlBuilder object that will run in
        ConvolutionVsContraction mode.
        """
        obj = cls.ConvolutionContraction(conv, {}, solution, problemFunc=cls.ProblemSizes, problemLevel=2, dataType=dataType)
        obj.doc["GlobalParameters"]["ConvolutionVsContraction"] = 1
        for problem in obj.doc["BenchmarkProblems"]:
            problem[0]["OperationType"] = conv.convolutionType
            problem[0]["ConvolutionConfig"] = [copy.deepcopy(conv.config)]

        return obj

    @classmethod
    def ConvolutionContraction(cls, conv, problemType, solution, dataType, \
                                problemFunc, generateConvFormat=True, problemLevel=1):
        """
        Generates a YamlBuilder object that will run a convolution, in normal
        contraction mode.
        """

        availableArchs = cls.findAvailableArchs()
        benchmarkParams = solution()

        for item in benchmarkParams["ForkParameters"]:
          if 'MatrixInstruction' in item and 'gfx908' not in availableArchs:
            pytest.skip()
            break

        doc = cls.Header(debug=False)

        if generateConvFormat:
            tensileProblemType = {
                "OperationType": conv.convolutionType,
                "ConvolutionConfig": [{key:val} for (key,val) in conv.config.items()],
                "DataType": dataType
            }
        else:
            tensileProblemType = {
                "OperationType": "TensorContraction",
                "DataType": dataType
            }
            tensileProblemType.update(problemType)

        for (key,value) in conv.solutionParms.items():
            benchmarkParams["ForkParameters"].append({key:[value]})
        problems = problemFunc(conv, problemType, problemLevel)

        #print("problems:", problems)
        if generateConvFormat:
            convs = [ {"Conv": e} for e in problems]
            benchmarkParams["BenchmarkFinalParameters"] = [{"ProblemSizes": convs }]
        else:
            exacts = [{"Exact": dict(ConvProblem(p, conv).toExactDict())} for p in problems]
            benchmarkParams["BenchmarkFinalParameters"] = [{"ProblemSizes": exacts}]

        doc["BenchmarkProblems"] = [[tensileProblemType, benchmarkParams]]

        #print (doc)

        return cls(doc)

    # shortcuts for setting parameters in tests:
defaultSizes = pytest.param((YamlBuilder.ProblemSizes, 1), id="default_sizes")
resnetSizes  = pytest.param((YamlBuilder.ProblemSizesResNet,1),id="resnet")
inceptionSizes  = pytest.param((YamlBuilder.ProblemSizesInception,1),id="inception")
