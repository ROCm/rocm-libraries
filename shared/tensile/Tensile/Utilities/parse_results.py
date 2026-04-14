#!/usr/bin/python3

################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

import fileinput,csv
import argparse
from collections import OrderedDict
from functools import reduce
import operator

class DataRow:
    """ Single result row """
    def __init__(self, row):
        self.dataRow = row
        self.derivedFields = {}

    def __getitem__(self, key):
        return self.dataRow[key]

    def finalizeDerivedFields(self, solution):
        self.derivedFields['best-any'] = str(solution.maxPerfRow["gflops"])
        if args.ops_per_cu_per_cycle:
            peakMFlops = float(self["clock-sys"]) * float(self["num-cus"]) * args.ops_per_cu_per_cycle
            sizes = [int(p) for p in self['problem-sizes'].strip("()").split(",")]
            work = reduce(operator.mul, sizes, 2.0)
            self.derivedFields['model-alu-us'] = '%6.0f' % (work / peakMFlops * float(self["total-gran"]))
        if args.mem_width:
            peakMBps = float(self["clock-mem"]) * args.mem_width * args.mem_eff/100.0
            self.derivedFields['model-mem-read-us']  = "%6.0f" % (float(self["mem-read-bytes"]) / peakMBps)
            self.derivedFields['model-mem-write-us'] = "%6.0f" % (float(self["mem-write-bytes"]) / peakMBps)


class Solution:
    """ Rows associated with this solution """
    def __init__(self):
        self.problemResults = []
        self.maxPerfRow = None

    def addRow(self,row):
        self.problemResults.append(row)
        if self.maxPerfRow==None or float(row["gflops"]) > float(self.maxPerfRow["gflops"]):
            self.maxPerfRow = row

class ProblemSize:
    """ Rows associated with this problem """
    def __init__(self):
        self.dataRows = []

    def addRow(self,row):
        self.dataRows.append(row)


class Reader:

    def formatCol(self, fieldName, val):
        return ('{0: <%d}'%self.fieldLength[fieldName]).format(val)

    def printRow(self, row, fieldsToPrint, derivedFieldsToPrint, end='\n'):
        for field in fieldsToPrint:
            print (self.formatCol(field, row[field]), end=" ")

        for field in derivedFieldsToPrint:
            print (self.formatCol(field, row.derivedFields[field]), end=" ")
        print (end=end)


    def readFile(self, fileName):
        db = 0

        csv_field_names = None
        csv_table=[]
        for line in fileinput.input(fileName):
            if line.startswith("run,"):
                csv_field_names = line.replace(' ', '').split(',')
            elif csv_field_names and "Contraction" in line:
                csv_table.append(line)
                if db:
                    print ("L=",line),

        dictReader = csv.DictReader(csv_table, csv_field_names, skipinitialspace=True)

        fieldsToPrint= ('problem-progress', 'problem-sizes', 'solution', 'time-us', 'gflops', \
                        'tiles-per-cu', 'total-gran', 'cu-gran', 'tile0-gran', 'tile1-gran')
        derivedFieldsToPrint = []
        if args.ops_per_cu_per_cycle:
            derivedFieldsToPrint += ["model-alu-us"]
        if args.mem_width:
            derivedFieldsToPrint += ["model-mem-read-us", "model-mem-write-us"]
        derivedFieldsToPrint += ['best-any']

        for f in fieldsToPrint:
            assert f not in derivedFieldsToPrint

        self.fieldLength={}
        for k in fieldsToPrint:
            self.fieldLength[k] = len(k) # length of name
        for k in derivedFieldsToPrint:
            self.fieldLength[k] = len(k)

        self.dataOut=[]
        self.solutions = {} # map from solution to Solution class
        problemSizes = OrderedDict()
        for row in dictReader:
            self.solutions.setdefault(row["solution"],Solution()).addRow(DataRow(row))
            problemSizes.setdefault(row["problem-sizes"],ProblemSize()).addRow(DataRow(row))

            keep = {}
            for k in fieldsToPrint:
                keep[k] = row[k]
                self.fieldLength[k] = max(len(row[k]), self.fieldLength[k])

            self.dataOut.append((keep))

        # after everything added, compute derived stats:
        for key,ps in problemSizes.items():
            for row in ps.dataRows:
                row.finalizeDerivedFields(self.solutions[row["solution"]])
                for k in derivedFieldsToPrint:
                    self.fieldLength[k] = max(len(row.derivedFields[k]), self.fieldLength[k])

        # sort by gflops
        self.dataOut.sort(key = lambda row: int(row["gflops"]), reverse=True)

        # print header:
        for field in fieldsToPrint:
            print (self.formatCol(field,val=field), end=" ")
        for field in derivedFieldsToPrint:
            print (field,end=" ")
        print()

        if args.print_winners:
            for key,ps in problemSizes.items():
                if args.problem_progress==None or int(ps.dataRows[0]["problem-progress"].split('/')[0]) in args.problem_progress:
                    dataRows = [row for row in ps.dataRows if args.filter in row["solution"]]
                    sortedRows = sorted(dataRows, key = lambda row: int(row["gflops"]), reverse=True)
                    for i in range(min(len(sortedRows),args.print_winners) if args.print_winners>=0 else len(sortedRows)):
                        self.printRow(sortedRows[i], fieldsToPrint, derivedFieldsToPrint)

        if args.print_summary:
            for row in self.dataOut:
                self.printRow(row, fieldsToPrint, derivedFieldsToPrint)



my_parser = argparse.ArgumentParser(description='summarize and analyze tensile benchmark results')

my_parser.add_argument('--print_winners', '-w', type=int, default=1,
                       help='print top N winners for each problem.  -1 prints all')
my_parser.add_argument('--problem_progress', '-p', type=int, action='append',
                       help='show only specified problem.  Can specify multiple times.')
my_parser.add_argument('--print_summary', '-s', action='store_true',
                       help='print all results')
my_parser.add_argument('--filter', '-k', type=str, default="",
                       help='filter solutions, ie "GSU1".')
my_parser.add_argument('--ops-per-cu-per-cycle', '-o', type=int, default=0,
                       help='ops per cu per cycle. Typical values 64(vega20 f64), 128(vega20 f32), etc. used to compute AluUs')
my_parser.add_argument('--mem-width', '-m', type=int, default=1024,
                       help='mem bus width in bits. Typical values 1024 (vega20).  Used to compute Mem*Us')
my_parser.add_argument('--mem-eff', '-e', type=int, default=85, choices=range(1,100),
                       help='efficiency of memory bus.  Must be integer 1.100.')

my_parser.add_argument('input_file', help='file with tensile output to parse')

args = my_parser.parse_args()

r = Reader()
r.readFile(args.input_file)
