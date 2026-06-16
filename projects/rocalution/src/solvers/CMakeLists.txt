# ########################################################################
# Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

set(SOLVERS_SOURCES
  solvers/krylov/cg.cpp
  solvers/krylov/fcg.cpp
  solvers/krylov/cr.cpp
  solvers/krylov/bicgstab.cpp
  solvers/krylov/bicgstabl.cpp
  solvers/krylov/qmrcgstab.cpp
  solvers/krylov/gmres.cpp
  solvers/krylov/fgmres.cpp
  solvers/krylov/idr.cpp
  solvers/multigrid/base_multigrid.cpp
  solvers/multigrid/base_amg.cpp
  solvers/multigrid/multigrid.cpp
  solvers/multigrid/unsmoothed_amg.cpp
  solvers/multigrid/smoothed_amg.cpp
  solvers/multigrid/ruge_stueben_amg.cpp
  solvers/multigrid/pairwise_amg.cpp
  solvers/direct/inversion.cpp
  solvers/direct/lu.cpp
  solvers/direct/qr.cpp
  solvers/solver.cpp
  solvers/chebyshev.cpp
  solvers/mixed_precision.cpp
  solvers/preconditioners/preconditioner.cpp
  solvers/preconditioners/preconditioner_blockjacobi.cpp
  solvers/preconditioners/preconditioner_ai.cpp
  solvers/preconditioners/preconditioner_as.cpp
  solvers/preconditioners/preconditioner_multielimination.cpp
  solvers/preconditioners/preconditioner_saddlepoint.cpp
  solvers/preconditioners/preconditioner_blockprecond.cpp
  solvers/preconditioners/preconditioner_multicolored.cpp
  solvers/preconditioners/preconditioner_multicolored_gs.cpp
  solvers/preconditioners/preconditioner_multicolored_ilu.cpp
  solvers/iter_ctrl.cpp
)

set(SOLVERS_PUBLIC_HEADERS
  solvers/krylov/cg.hpp
  solvers/krylov/fcg.hpp
  solvers/krylov/cr.hpp
  solvers/krylov/bicgstab.hpp
  solvers/krylov/bicgstabl.hpp
  solvers/krylov/qmrcgstab.hpp
  solvers/krylov/gmres.hpp
  solvers/krylov/fgmres.hpp
  solvers/krylov/idr.hpp
  solvers/multigrid/base_multigrid.hpp
  solvers/multigrid/base_amg.hpp
  solvers/multigrid/multigrid.hpp
  solvers/multigrid/unsmoothed_amg.hpp
  solvers/multigrid/smoothed_amg.hpp
  solvers/multigrid/ruge_stueben_amg.hpp
  solvers/multigrid/pairwise_amg.hpp
  solvers/direct/inversion.hpp
  solvers/direct/lu.hpp
  solvers/direct/qr.hpp
  solvers/solver.hpp
  solvers/chebyshev.hpp
  solvers/mixed_precision.hpp
  solvers/preconditioners/preconditioner.hpp
  solvers/preconditioners/preconditioner_blockjacobi.hpp
  solvers/preconditioners/preconditioner_ai.hpp
  solvers/preconditioners/preconditioner_as.hpp
  solvers/preconditioners/preconditioner_multielimination.hpp
  solvers/preconditioners/preconditioner_saddlepoint.hpp
  solvers/preconditioners/preconditioner_blockprecond.hpp
  solvers/preconditioners/preconditioner_multicolored.hpp
  solvers/preconditioners/preconditioner_multicolored_gs.hpp
  solvers/preconditioners/preconditioner_multicolored_ilu.hpp
  solvers/iter_ctrl.hpp
)
