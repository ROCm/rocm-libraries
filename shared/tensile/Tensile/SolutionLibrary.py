################################################################################
#
# Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

import itertools
import re

from . import Properties
from . import Hardware
from . import Common
from . import Contractions
from .SolutionStructs import Solution as OriginalSolution
from .Utils import state

class SingleSolutionLibrary:
    Tag = "Single"

    # allow null solution. This enables the solution library to have
    # an effective default value when the related field in the logic
    # file is optional.
    def __init__(self, solution = None):
        self.solution = solution

    @property
    def tag(self):
        return self.__class__.Tag

    def state(self):
        # handles the case when the solution is null.
        # -1 index represents null solution.
        if self.solution:
           return {"type": self.tag, "index": self.solution.index}
        else:
           return {"type": self.tag, "index": -1}

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
        if distance == "Equality":
            propertyKeys[0] = Properties.Property("BatchSize", index=0)

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


class DecisionTreeLibrary:
    Tag = "DecisionTree"
    StateKeys = [("type", "tag"), "features", "trees", "nullValue"]

    @classmethod
    def FromOriginalState(cls, d, solutions):
        features = d["features"]
        origTrees = d["trees"]

        # Support for legacy implementation of the dtree which has no fallback.
        if "fallback" in d:
            fallbackIndex = d["fallback"]
            nullValue = SingleSolutionLibrary(solutions[fallbackIndex])
        else:
            # The default solution is null
            nullValue = SingleSolutionLibrary()

        trees = []

        for tree in origTrees:
            index = tree["solution"]
            value = SingleSolutionLibrary(solutions[index])

            entry = {"tree": tree["tree"], "value": value}
            trees.append(entry)

        return cls(features, trees, nullValue)

    @property
    def tag(self):
        return self.__class__.Tag

    def merge(self, other):
        raise RuntimeError(
            "DecisionTreeLibrary does not support merging; ensure each library row has a unique predicate"
        )

    def remapSolutionIndices(self, indexMap):
        pass

    def __init__(self, features, trees, nullValue):
        self.features = features
        self.trees = trees
        self.nullValue = nullValue


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

        # Sort to ensure consistent fallback logic.
        self.rows.sort(key=lambda x: x["predicate"])

    def remapSolutionIndices(self, indexMap):
        for row in self.rows:
            row["library"].remapSolutionIndices(indexMap)


class MasterSolutionLibrary:
    StateKeys = ["solutions", "library"]
    ArchitectureSet = set()

    @classmethod
    def ArchitectureIndexMap(cls, architectureName: str) -> int:
        """Maps hex characters from gfx name to an index.

        Given a gfx name of the form gfx[0-9a-f]*, map the characters following
        gfx from hex to int and left shift the integer by 18.

        Args:
            architectureName: The gfx name (or fallback).

        Returns:
            An integer representing the index for the given gfx architecture.

        Raises:
            RunTimeError: The provided architecture was already mapped to an index or umappable.
        """
        archval = -1
        if architectureName == "fallback":
            archval = 0
        elif architectureName.startswith("gfx"):
            archString = re.search('(?<=gfx)[0-9a-f]*', architectureName)
            if archString is not None:
                archLiteral = archString.group(0)
                archval = (int(archLiteral, 16) << 18)
        # Check for duplicate architecture values
        if archval >= 0 and not archval in cls.ArchitectureSet:
            cls.ArchitectureSet.add(archval)
        else:
            raise RuntimeError("ERROR in architecture solution index mapping.")
        return archval

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
                          solutionClass=Contractions.Solution,
                          libraryOrder=None):

        # functions for creating each "level" of the library
        def hardware(d, problemType, solutions, library, placeholderName):
            devicePart = d["ArchitectureName"]
            cuCount = d["CUCount"]
            isAPU = d["IsAPU"]

            newLib = PredicateLibrary(tag="Hardware")
            if devicePart == "fallback":
                pred = Hardware.HardwarePredicate("TruePred")
            else:
                pred = Hardware.HardwarePredicate.FromHardware(Common.gfxArch(devicePart), cuCount, isAPU)

            newLib.rows.append({"predicate": pred, "library": library})

            if lazyLibrary:
                if cuCount:
                    placeholderName += "_CU" + str(cuCount)
                if isAPU is not None:
                    placeholderName += "_APU" if isAPU == 1 else "_XPU"
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

        def fp16AltImpl(d, problemType, solutions, library, placeholderName):
            if d.get("Fp16AltImpl"):
                predicate = Properties.Predicate(tag="Fp16AltImpl")
            else:
                predicate = Properties.Predicate(tag="TruePred")
            newLib = PredicateLibrary(tag="Problem")
            newLib.rows.append({"predicate": predicate, "library": library})

            if lazyLibrary and predicate.tag == "Fp16AltImpl" and not d.get("Fp16AltImplRnz"):
                placeholderName += "_Fp16Alt"

            return newLib, placeholderName

        def fp16AltImplRound(d, problemType, solutions, library, placeholderName):
            if d.get("Fp16AltImplRound"):
                predicate = Properties.Predicate(tag="Fp16AltImplRound")
            else:
                predicate = Properties.Predicate(tag="TruePred")
            newLib = PredicateLibrary(tag="Problem")
            newLib.rows.append({"predicate": predicate, "library": library})

            if lazyLibrary and predicate.tag == "Fp16AltImplRound":
                placeholderName += "_Fp16AltRnz"

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
                else:
                    predicate = Properties.Predicate(tag="TruePred")

                matchingLib = MatchingLibrary.FromOriginalState(d["Library"], solutions)
                library = PredicateLibrary(tag="Problem")
                library.rows.append({"predicate": predicate, "library": matchingLib})

            elif d["LibraryType"] == "DecisionTree":
                library = PredicateLibrary(tag="Problem")
                for lib in d["Library"]:
                    preds = lib["region"]
                    predObjs = [Properties.Predicate.FromOriginalState(p) for p in preds]

                    if len(predObjs) == 1:
                        predicate = predObjs[0]
                    else:
                        predicate = Properties.Predicate.And(predObjs)

                    treeLib = DecisionTreeLibrary.FromOriginalState(lib, solutions)
                    library.rows.append({"predicate": predicate, "library": treeLib})

            return library, placeholderName

        # end library creation functions

        if libraryOrder is None:
            if Common.globalParameters["LazyLibraryLoading"]:
                libraryOrder = [
                    hardware, operationIdentifier, performanceMetric, fp16AltImpl, fp16AltImplRound, predicates,
                    placeholder, selection
                ]
            else:
                libraryOrder = [
                    hardware, operationIdentifier, performanceMetric, fp16AltImpl, fp16AltImplRound, predicates,
                    selection
                ]
        #assert libraryOrder[-1] == selection

        lazyLibrary = None
        if placeholder in libraryOrder:
            placeholderIndex = libraryOrder.index(placeholder) + 1
            lazyLibrary = MasterSolutionLibrary.FromOriginalState(origData, origSolutions,
                                                                  solutionClass,
                                                                  libraryOrder[placeholderIndex:])
            libraryOrder = libraryOrder[0:placeholderIndex]
            origSolutions = []

        problemType = Contractions.ProblemType.FromOriginalState(origData["ProblemType"])
        allSolutions = [solutionClass.FromSolutionStruct(s) for s in origSolutions]
        cls.FixSolutionIndices(allSolutions)

        # library is constructed in reverse order i.e. bottom-up
        library = None
        placeholderName = "TensileLibrary"
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

        return rv

    @classmethod
    def BenchmarkingLibrary(cls, solutions):
        solutionObjs = list([Contractions.Solution.FromOriginalState(s.getAttributes()) for s in solutions])
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
            "solutions": state(iter(list(self.solutions.values()))),
            "library": state(self.library)
        }

        if self.version is not None:
            rv["version"] = self.version
        return rv

    def applyNaming(self, naming=None):
        if naming is None:
            #allSolutions = itertools.chain(iter(list(self.solutions.values())), iter(list(self.sourceSolutions.values())))
            kernels = list(
                itertools.chain(*[s.originalSolution.getKernels()
                                  for s in self.solutions.values()]))
            naming = OriginalSolution.getMinNaming(kernels)

        for s in list(self.solutions.values()):
            s.name = OriginalSolution.getNameMin(s.originalSolution.getKernels(), naming)

    def _remapSolutionIndicesStartingFrom(self, library, solutions: dict, curIndex: int) -> tuple:
        reIndexMap = {}
        newSolutions = {}
        for _, soln in solutions.items():
            reIndexMap[soln.index] = curIndex
            soln.index = curIndex
            newSolutions[curIndex] = soln
            curIndex += 1
        return newSolutions, reIndexMap

    def remapSolutionIndicesStartingFrom(self, startingIndex) -> None:
        """Remap all the solution indexes for a given library.

        Given a starting index, remap all indexes for a given library
        including the lazy libraries if enabled. If a library has five
        solutions and the starting index is 222, the solution.index fields
        would be 222, 223, 224, 225 and 226 respectively. If there are two
        lazy libraries and each have two solutions, the solution.index
        fields would be 222, 223, 224 and 225 respectively.

        Args:
            startingIndex: The index to start remapping from.
        """
        if self.lazyLibraries:
            lazyLibrary = {}
            for name, lib in self.lazyLibraries.items():
                lib.solutions, reIndexMap = self._remapSolutionIndicesStartingFrom(lib.library, lib.solutions, startingIndex)
                lib.library.remapSolutionIndices(reIndexMap)
                lazyLibrary[name] = lib
            self.lazyLibraries = lazyLibrary

        self.solutions, reIndexMap =  self._remapSolutionIndicesStartingFrom(self.library, self.solutions, startingIndex)
        self.library.remapSolutionIndices(reIndexMap)

    def insert(self, other):
        assert self.__class__ == other.__class__

        for name, lib in other.lazyLibraries.items():
            self.lazyLibraries[name] = lib

        for _, s in other.solutions.items():
            self.solutions[s.index] = s
        self.library.merge(other.library)

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
        return "SolutionLibrary<ContractionProblem, ContractionSolution>"

    @property
    def cpp_class(self):
        return "MasterSolutionLibrary<ContractionProblem, ContractionSolution>"
