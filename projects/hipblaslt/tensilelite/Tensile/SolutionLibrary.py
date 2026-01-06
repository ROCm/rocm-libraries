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

from typing import Dict

from . import Properties
from . import Hardware
from . import Contractions
from Tensile.Common import state, IsaInfo
from Tensile.Common.Architectures import gfxToIsa
from Tensile.SolutionStructs.Naming import getSolutionNameMin, getKernelNameMin

class SingleSolutionLibrary:
    Tag = "Single"

    def __init__(self, solution):
        self.solution = solution

    @property
    def tag(self):
        return self.__class__.Tag

    def state(self):
        return {"type": self.tag, "index": self.solution.index}

    def remapSolutionIndices(self, indexMap):
        pass

class IndexSolutionLibrary(SingleSolutionLibrary):
    Tag = "Index"

    def state(self):
        return self.solution.index


class PlaceholderLibrary:
    Tag = 'Placeholder'

    def __init__(self, name):
        self.filenamePrefix = name

    @property
    def tag(self):
        return self.__class__.Tag

    def state(self):
        return {'type': self.tag, 'value': self.filenamePrefix}

    def remapSolutionIndices(self, indexMap):
        pass

    def merge(self, other):
        pass


class MatchingLibrary:
    Tag = "Matching"
    StateKeys = [("type", "tag"), "properties", "table", "distance"]

    @classmethod
    def FromOriginalState(cls, d, solutions):
        indices = d["indexOrder"]
        distance = d["distance"]
        origTable = d["table"]

        propertyKeys = {
            2: Properties.Property("FreeSizeA", index=0),
            3: Properties.Property("FreeSizeB", index=0),
            1: Properties.Property("BoundSize", index=0)
        }
        if distance == "Equality" or distance == "GridBased":
            propertyKeys[0] = Properties.Property("BatchSize", index=0)

        if distance == "Range":
            propertyKeys = {
                0: Properties.Property("FreeSizeA", index=0),
                1: Properties.Property("FreeSizeA", index=0),
                2: Properties.Property("FreeSizeB", index=0),
                3: Properties.Property("FreeSizeB", index=0),
                4: Properties.Property("BatchSize", index=0),
                5: Properties.Property("BatchSize", index=0),
                6: Properties.Property("BoundSize", index=0),
                7: Properties.Property("BoundSize", index=0)
            }

        properties = list([propertyKeys[i] for i in indices if i in propertyKeys])
        keyOrder = [i for i, j in enumerate(indices) if j in propertyKeys]

        table = []

        for row in origTable:
            try:
                index = row[1][0]
                value = IndexSolutionLibrary(solutions[index])
                key = list([row[0][i] for i in keyOrder])
                if distance == "GridBased":
                    entry = {"key": key, "index": value}
                else:
                    entry = {"key": key, "index": value, "speed": row[1][1]}

                table.append(entry)
            except KeyError:
                pass

        table.sort(key=lambda r: r["key"])

        return cls(properties, table, distance)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        assert self.__class__ == other.__class__ \
                and self.properties == other.properties \
                and self.distance == other.distance

        self.table += other.table

        self.table.sort(key=lambda r: r["key"])

    def remapSolutionIndices(self, indexMap):
        pass

    def __init__(self, properties, table, distance):
        self.properties = properties
        self.table = table
        self.distance = distance

class FreeSizeLibrary:
    Tag = "FreeSize"
    StateKeys = [("type", "tag"), "table"]

    @classmethod
    def FromOriginalState(cls, d, solutions):
        origTable = d["table"]

        table = []

        try:
            indexStart  = origTable[0]
            indexOffset = origTable[1]
            for index in range(indexStart, indexStart + indexOffset):
                value = IndexSolutionLibrary(solutions[index])
                entry = {"index": value}
                table.append(entry)
        except KeyError:
            pass

        return cls(table)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        assert self.__class__ == other.__class__

        self.table += other.table

    def remapSolutionIndices(self, indexMap):
        pass

    def __init__(self, table):
        self.table = table

class PredictionLibrary:
    Tag = "Prediction"
    StateKeys = [("type", "tag"), "table"]

    @classmethod
    def FromOriginalState(cls, d, solutions):
        origTable = d["table"]

        table = []

        try:
            indexStart  = origTable[0]
            indexOffset = origTable[1]
            for index in range(indexStart, indexStart + indexOffset):
                value = IndexSolutionLibrary(solutions[index])
                table.append(value)
        except KeyError:
            pass

        return cls(table)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        assert self.__class__ == other.__class__

        self.table += other.table

    def remapSolutionIndices(self, indexMap):
        pass

    def __init__(self, table):
        self.table = table

class MLPClassificationLibrary:
    Tag = "MLPClassification"
    StateKeys = [("type", "tag"), "table", "mlp", "problemFeatures"]

    @classmethod
    def FromOriginalState(cls, d, solutions):
        origTable = d["table"]
        table = []

        try:
            indexStart  = origTable[0]
            indexOffset = origTable[1]
            for index in range(indexStart, indexStart + indexOffset):
                value = IndexSolutionLibrary(solutions[index])
                table.append(value)
        except KeyError:
            pass

        mlp = d["mlp"]
        problem_features = d["problemFeatures"]
        return cls(table, mlp, problem_features)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        raise RuntimeError(
            "MLPClassificationLibrary does not support merging."
        )

    def remapSolutionIndices(self, indexMap):
        pass

    def __init__(self, table, mlp, problem_features):
        self.table = table
        self.mlp = mlp
        self.problemFeatures = problem_features


class ProblemMapLibrary:
    Tag = "ProblemMap"
    StateKeys = [("type", "tag"), ("property", "mappingProperty"), ("map", "mapping")]

    def __init__(self, mappingProperty=None, mapping=None):
        self.mappingProperty = mappingProperty
        self.mapping = mapping

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        assert self.__class__ == other.__class__ and self.tag == other.tag and self.mappingProperty == other.mappingProperty

        for key, value in list(other.mapping.items()):
            if key in self.mapping:
                self.mapping[key].merge(value)
            else:
                self.mapping[key] = value

    def remapSolutionIndices(self, indexMap):
        for key, value in list(self.mapping.items()):
            value.remapSolutionIndices(indexMap)


class PredicateLibrary:
    StateKeys = [("type", "tag"), "rows"]

    def __init__(self, tag=None, rows=None):
        self.tag = tag
        if rows is None: rows = []
        self.rows = rows

    def merge(self, other):
        assert self.__class__ == other.__class__ and self.tag == other.tag

        rowPreds = [r["predicate"] for r in self.rows]

        for row in other.rows:
            if row["predicate"] in rowPreds:
                myRownum = rowPreds.index(row["predicate"])
                self.rows[myRownum]["library"].merge(row["library"])
            else:
                self.rows.append(row)

        if self.rows[0]["library"].tag == "Placeholder":
            # Sort to ensure pure gemm can be search first
            self.rows.sort(key=lambda x: len(x["library"].filenamePrefix))
        else:
            # Sort to ensure consistent fallback logic.
            self.rows.sort(key=lambda x: x["predicate"])

    def remapSolutionIndices(self, indexMap):
        for row in self.rows:
            row["library"].remapSolutionIndices(indexMap)


class MasterSolutionLibrary:
    StateKeys = ["solutions", "library"]

    @classmethod
    def FixSolutionIndices(cls, solutions):
        # fix missing and duplicate solution indices.
        try:
            maxSolutionIdx = max([s.index for s in solutions if s.index is not None])
        except ValueError:
            maxSolutionIdx = -1

        solutionsSoFar = set()
        for solution in solutions:
            if solution.index is None or solution.index in solutionsSoFar:
                maxSolutionIdx += 1
                solution.index = maxSolutionIdx
            else:
                solutionsSoFar.add(solution.index)

    @classmethod
    def FromOriginalState(cls,
                          origData,
                          origSolutions,
                          splitGSU: bool,
                          printSolutionRejectionReason: bool,
                          printIndexAssignmentInfo: bool,
                          assembler,
                          isaInfoMap: Dict[str, IsaInfo],
                          lazyLibraryLoading: bool,
                          solutionClass=Contractions.Solution,
                          libraryOrder=None,
                          placeholderName='TensileLibrary'):

        # functions for creating each "level" of the library
        def hardware(d, problemType, solutions, library, placeholderName):
            devicePart = d["ArchitectureName"]
            cuCount = d["CUCount"]

            newLib = PredicateLibrary(tag="Hardware")
            if devicePart == "fallback":
                pred = Hardware.HardwarePredicate("TruePred")
            else:
                pred = Hardware.HardwarePredicate.FromHardware(gfxToIsa(devicePart), cuCount)

            newLib.rows.append({"predicate": pred, "library": library})

            if lazyLibrary:
                if cuCount: placeholderName += "_CU" + str(cuCount)
                placeholderName += "_" + str(devicePart)

            return newLib, placeholderName

        def operationIdentifier(d, problemType, solutions, library, placeholderName):
            operationID = problemType.operationIdentifier
            prop = Properties.Property("OperationIdentifier")
            mapping = {operationID: library}

            newLib = ProblemMapLibrary(prop, mapping)

            if lazyLibrary:
                placeholderName += "_" + operationID

            return newLib, placeholderName

        def performanceMetric(d, problemType, solutions, library, placeholderName):
            if d.get("PerfMetric", "DeviceEfficiency") != "DeviceEfficiency":
                predicate = Properties.Predicate(tag=d["PerfMetric"])
            else:
                predicate = Properties.Predicate(tag="TruePred")
            newLib = PredicateLibrary(tag="Problem")
            newLib.rows.append({"predicate": predicate, "library": library})

            if lazyLibrary and predicate.tag != "TruePred":
                placeholderName += "_" + predicate.tag

            return newLib, placeholderName

        def predicates(d, problemType, solutions, library, placeholderName):
            predicates = problemType.predicates(includeBatch=True, includeType=True)
            predicate = Contractions.ProblemPredicate.And(predicates)

            newLib = PredicateLibrary(tag="Problem")
            newLib.rows.append({"predicate": predicate, "library": library})

            if lazyLibrary:
                placeholderName += problemType.placeholderStr(includeBatch=True, includeType=True)

            return newLib, placeholderName

        def placeholder(d, problemType, solutions, library, placeholderName):
            newLib = PlaceholderLibrary(placeholderName)
            return newLib, placeholderName

        def selection(d, problemType, solutions, library, placeholderName):
            if d["LibraryType"] == "Matching":
                if d["Library"]["distance"] == "Equality":
                    predicate = Properties.Predicate(tag="EqualityMatching")
                elif d["Library"]["distance"] == "Range":
                    predicate = Properties.Predicate(tag="RangeMatching")
                else:
                    predicate = Properties.Predicate(tag="TruePred")

                matchingLib = MatchingLibrary.FromOriginalState(d["Library"], solutions)
                library = PredicateLibrary(tag="Problem")
                library.rows.append({"predicate": predicate, "library": matchingLib})
            elif d["LibraryType"] == "FreeSize":
                predicate = Properties.Predicate(tag="FreeSizeMatching")

                freesizeLib = FreeSizeLibrary.FromOriginalState(d["Library"], solutions)
                library = PredicateLibrary(tag="Problem")
                library.rows.append({"predicate": predicate, "library": freesizeLib})
            elif d["LibraryType"] == "Prediction":
                predicate = Properties.Predicate(tag="FreeSizeMatching") # TODO Do we need a new predicate here?

                predictionLib = PredictionLibrary.FromOriginalState(d["Library"], solutions)
                library = PredicateLibrary(tag="Problem")
                library.rows.append({"predicate": predicate, "library": predictionLib})
            elif d["LibraryType"] == "MLPClassification":
                predicate = Properties.Predicate(tag="TruePred")

                regressionLib = MLPClassificationLibrary.FromOriginalState(d["Library"], solutions)
                library = PredicateLibrary(tag="Problem")
                library.rows.append({"predicate": predicate, "library": regressionLib})
            else:
                assert 0 and "Unrecognized LibraryType."

            if lazyLibraryLoading:
                placeholderName += '_' + str(problemType.aType) + str(problemType.bType)
                placeholderName += '_' + str(problemType.cType) + str(problemType.computeInputType)
                if problemType.activationType != 'none':
                    if str(problemType.activationType).upper() == 'ALL':
                        placeholderName += "_A"
                    elif str(problemType.activationType).upper() == 'HIPBLASLT_ALL':
                        placeholderName += "_HA"
                    else:
                        placeholderName += "_%s"%str(problemType.activationType).upper()

                if problemType.swizzleTensorA:
                    placeholderName += '_STA'

                if problemType.swizzleTensorB:
                    placeholderName += '_STB'

                if problemType.useBias:
                    placeholderName += '_Bias'
                if problemType.useE:
                    placeholderName += '_Grad' if problemType.useGradient else '_Aux'
                if problemType.groupedGemm:
                    placeholderName += "_GG"
                else:
                    placeholderName += "" if problemType.stridedBatched else "_GB" # legacy
                if problemType.useScaleAB == "Scalar":
                    placeholderName += '_SAB'
                elif problemType.useScaleAB == "Vector":
                    placeholderName += '_SABV'
                if problemType.useScaleCD:
                    placeholderName += '_SCD'
                if problemType.useScaleAlphaVec:
                    placeholderName += '_SAV'
                if problemType.sparse:
                    placeholderName += '_SPB' if problemType.sparse == 2 else '_SPA'
                if not problemType.f32XdlMathOp.isSingle() and problemType.computeInputType.isSingle():
                    placeholderName += '_M' + str(problemType.f32XdlMathOp)
                if problemType.supportDeviceUserArguments:
                    placeholderName += '_UA'

            return library, placeholderName

        # end library creation functions

        if libraryOrder is None:
            if lazyLibraryLoading:
                libraryOrder = [
                    hardware, operationIdentifier, performanceMetric, predicates,
                    placeholder, selection
                ]
            else:
                libraryOrder = [
                    hardware, operationIdentifier, performanceMetric, predicates,
                    selection
                ]
        #assert libraryOrder[-1] == selection

        lazyLibrary = None
        if placeholder in libraryOrder:
            placeholderIndex = libraryOrder.index(placeholder) + 1
            lazyLibrary, placeholderName = \
                MasterSolutionLibrary.FromOriginalState(origData,
                                                        origSolutions,
                                                        splitGSU,
                                                        printSolutionRejectionReason,
                                                        printIndexAssignmentInfo,
                                                        assembler,
                                                        isaInfoMap,
                                                        lazyLibraryLoading,
                                                        solutionClass,
                                                        libraryOrder[placeholderIndex:],
                                                        placeholderName)
            libraryOrder = libraryOrder[0:placeholderIndex]
            origSolutions = []

        problemType = Contractions.ProblemType.FromOriginalState(origData["ProblemType"])
        allSolutions = [solutionClass.FromSolutionStruct(
                            s,
                            splitGSU,
                            printSolutionRejectionReason,
                            printIndexAssignmentInfo,
                            assembler,
                            isaInfoMap
                        ) for s in origSolutions]
        cls.FixSolutionIndices(allSolutions)

        # library is constructed in reverse order i.e. bottom-up
        library = None
        placeholderLibrary = None
        for libName in reversed(libraryOrder):
            library, placeholderName = libName(origData, problemType, allSolutions, library,
                                               placeholderName)
            if libName == placeholder:
                placeholderLibrary = library

        solutions = {s.index: s for s in allSolutions}
        rv = cls(solutions, library)
        if lazyLibrary and placeholderLibrary:
            rv.lazyLibraries[placeholderName] = lazyLibrary
            placeholderLibrary.filenamePrefix = placeholderName

        return rv, placeholderName

    @classmethod
    def BenchmarkingLibrary(
        cls,
        solutions,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap
    ):
        solutionObjs = list([Contractions.Solution.FromOriginalState(
                                 s._state,
                                 splitGSU,
                                 printSolutionRejectionReason,
                                 printIndexAssignmentInfo,
                                 assembler,
                                 isaInfoMap)
                            for s in solutions])
        cls.FixSolutionIndices(solutionObjs)

        predRows = list([{
            "predicate": s.problemPredicate,
            "library": SingleSolutionLibrary(s)
        } for s in solutionObjs])
        library = PredicateLibrary(tag="Problem", rows=predRows)

        solutionMap = {s.index: s for s in solutionObjs}

        return cls(solutionMap, library)

    def __init__(self, solutions, library, version=None):
        self.lazyLibraries = {}
        self.solutions = solutions
        self.library = library
        self.version = version

    def state(self):
        rv = {
            "solutions": state(self.solutions.values()),
            "library": state(self.library)
        }

        if self.version is not None:
            rv["version"] = self.version
        return rv

    def applyNaming(self, splitGSU: bool):
        for s in list(self.solutions.values()):
            s.name = getSolutionNameMin(s.originalSolution.getKernels()[0], splitGSU)
            s.kernelName = getKernelNameMin(s.originalSolution.getKernels()[0], splitGSU)

    def remapSolutionIndicesStartingFrom(self, curIndex):
        reIndexMap = {}
        solutionCopy = self.solutions
        self.solutions = dict()
        for k, s in solutionCopy.items():
            reIndexMap[s.index] = curIndex
            s.index = curIndex
            self.solutions[curIndex] = s
            curIndex += 1

        self.library.remapSolutionIndices(reIndexMap)

    def merge(self, other, startIndex=0):
        assert self.__class__ == other.__class__

        curIndex = max(startIndex, max(self.solutions.keys()) + 1 if self.solutions else 0)
        if self.lazyLibraries:
            curIndex = max(
                curIndex,
                max(max(lib.solutions.keys()) for _, lib in self.lazyLibraries.items()) + 1)

        #Merge separate library files
        for name, lib in other.lazyLibraries.items():
            if name in self.lazyLibraries.keys():
                curIndex = self.lazyLibraries[name].merge(lib, curIndex)
            else:
                reIndexMap = {}
                newSolutions = {}

                for k, s in lib.solutions.items():
                    reIndexMap[s.index] = curIndex
                    s.index = curIndex
                    newSolutions[curIndex] = s
                    curIndex += 1

                lib.solutions = newSolutions
                lib.library.remapSolutionIndices(reIndexMap)

                self.lazyLibraries[name] = lib

        reIndexMap = {}
        for k, s in other.solutions.items():
            reIndexMap[s.index] = curIndex
            s.index = curIndex
            self.solutions[curIndex] = s
            curIndex += 1

        other.library.remapSolutionIndices(reIndexMap)

        self.library.merge(other.library)

        return curIndex  #Next unused index

    @property
    def cpp_base_class(self):
        return "SolutionLibrary<ContractionProblemGemm, ContractionSolution>"

    @property
    def cpp_class(self):
        return "MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>"
