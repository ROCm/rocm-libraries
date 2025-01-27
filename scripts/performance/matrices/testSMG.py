from sparseMatrixGenerator import *

A = fullyRandom(5, 5, 6)
convert('testM0', A)

A = (fullyRandom(5, 5, 6) + consecutiveCols(5, 5, 3, 2).transpose()).transpose()
convert('testM1', A)
