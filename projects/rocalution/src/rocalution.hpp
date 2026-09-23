/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief rocalution.hpp includes other *.hpp and exposes a common interface
 */

#ifndef ROCALUTION_ROCALUTION_HPP_
#define ROCALUTION_ROCALUTION_HPP_

#include "version.hpp"

#include "base/backend_manager.hpp"
#include "base/parallel_manager.hpp"

#include "base/operator.hpp"
#include "base/vector.hpp"

#include "base/global_matrix.hpp"
#include "base/local_matrix.hpp"
#include "base/matrix_formats.hpp"

#include "base/global_vector.hpp"
#include "base/local_vector.hpp"

#include "base/local_stencil.hpp"
#include "base/stencil_types.hpp"

#include "solvers/chebyshev.hpp"
#include "solvers/direct/inversion.hpp"
#include "solvers/direct/lu.hpp"
#include "solvers/direct/qr.hpp"
#include "solvers/iter_ctrl.hpp"
#include "solvers/krylov/bicgstab.hpp"
#include "solvers/krylov/bicgstabl.hpp"
#include "solvers/krylov/cg.hpp"
#include "solvers/krylov/cr.hpp"
#include "solvers/krylov/fcg.hpp"
#include "solvers/krylov/fgmres.hpp"
#include "solvers/krylov/gmres.hpp"
#include "solvers/krylov/idr.hpp"
#include "solvers/krylov/qmrcgstab.hpp"
#include "solvers/mixed_precision.hpp"
#include "solvers/multigrid/base_amg.hpp"
#include "solvers/multigrid/base_multigrid.hpp"
#include "solvers/multigrid/multigrid.hpp"
#include "solvers/multigrid/pairwise_amg.hpp"
#include "solvers/multigrid/ruge_stueben_amg.hpp"
#include "solvers/multigrid/smoothed_amg.hpp"
#include "solvers/multigrid/unsmoothed_amg.hpp"
#include "solvers/solver.hpp"

#include "solvers/preconditioners/preconditioner.hpp"
#include "solvers/preconditioners/preconditioner_ai.hpp"
#include "solvers/preconditioners/preconditioner_as.hpp"
#include "solvers/preconditioners/preconditioner_blockjacobi.hpp"
#include "solvers/preconditioners/preconditioner_blockprecond.hpp"
#include "solvers/preconditioners/preconditioner_multicolored.hpp"
#include "solvers/preconditioners/preconditioner_multicolored_gs.hpp"
#include "solvers/preconditioners/preconditioner_multicolored_ilu.hpp"
#include "solvers/preconditioners/preconditioner_multielimination.hpp"
#include "solvers/preconditioners/preconditioner_saddlepoint.hpp"

#include "utils/allocate_free.hpp"
#include "utils/time_functions.hpp"
#include "utils/types.hpp"

#endif // ROCALUTION_ROCALUTION_HPP_
