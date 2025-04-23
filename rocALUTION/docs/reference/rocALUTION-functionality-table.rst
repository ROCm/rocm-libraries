.. meta::
   :description: rocALUTION functionality table
   :keywords: rocALUTION, ROCm, library, API, functionality table
   
.. _functionality-table:

*******************
Functionality table
*******************

The following tables list the rocALUTION routines along with the information about the implementation location i.e. host backend, accelerator backend, or both.

LocalMatrix and LocalVector classes
===================================

All matrix operations (except SpMV) require a CSR matrix.

.. note:: 

   If the input matrix is not a CSR matrix, an internal conversion is performed to CSR format, followed by a back conversion to the previous format after the operation. In this case, a warning message on verbosity level 2 is printed.

==================================================================================== =============================================================================== ======== =======
**LocalMatrix function**                                                             **Comment**                                                                     **Host** **HIP**
==================================================================================== =============================================================================== ======== =======
:cpp:func:`GetFormat <rocalution::LocalMatrix::GetFormat>`                           Obtain the matrix format                                                        Yes      Yes
:cpp:func:`Check <rocalution::LocalMatrix::Check>`                                   Check the matrix for structure and value validity                               Yes      No
:cpp:func:`AllocateCSR <rocalution::LocalMatrix::AllocateCSR>`                       Allocate CSR matrix                                                             Yes      Yes
:cpp:func:`AllocateBCSR <rocalution::LocalMatrix::AllocateBCSR>`                     Allocate BCSR matrix                                                            Yes      Yes
:cpp:func:`AllocateMCSR <rocalution::LocalMatrix::AllocateMCSR>`                     Allocate MCSR matrix                                                            Yes      Yes
:cpp:func:`AllocateCOO <rocalution::LocalMatrix::AllocateCOO>`                       Allocate COO matrix                                                             Yes      Yes
:cpp:func:`AllocateDIA <rocalution::LocalMatrix::AllocateDIA>`                       Allocate DIA matrix                                                             Yes      Yes
:cpp:func:`AllocateELL <rocalution::LocalMatrix::AllocateELL>`                       Allocate ELL matrix                                                             Yes      Yes
:cpp:func:`AllocateHYB <rocalution::LocalMatrix::AllocateHYB>`                       Allocate HYB matrix                                                             Yes      Yes
:cpp:func:`AllocateDENSE <rocalution::LocalMatrix::AllocateDENSE>`                   Allocate DENSE matrix                                                           Yes      Yes
:cpp:func:`SetDataPtrCSR <rocalution::LocalMatrix::SetDataPtrCSR>`                   Initialize matrix with externally allocated CSR data                            Yes      Yes
:cpp:func:`SetDataPtrMCSR <rocalution::LocalMatrix::SetDataPtrMCSR>`                 Initialize matrix with externally allocated MCSR data                           Yes      Yes
:cpp:func:`SetDataPtrCOO <rocalution::LocalMatrix::SetDataPtrCOO>`                   Initialize matrix with externally allocated COO data                            Yes      Yes
:cpp:func:`SetDataPtrDIA <rocalution::LocalMatrix::SetDataPtrDIA>`                   Initialize matrix with externally allocated DIA data                            Yes      Yes
:cpp:func:`SetDataPtrELL <rocalution::LocalMatrix::SetDataPtrELL>`                   Initialize matrix with externally allocated ELL data                            Yes      Yes
:cpp:func:`SetDataPtrDENSE <rocalution::LocalMatrix::SetDataPtrDENSE>`               Initialize matrix with externally allocated DENSE data                          Yes      Yes
:cpp:func:`LeaveDataPtrCSR <rocalution::LocalMatrix::LeaveDataPtrCSR>`               Direct Memory access                                                            Yes      Yes
:cpp:func:`LeaveDataPtrMCSR <rocalution::LocalMatrix::LeaveDataPtrMCSR>`             Direct Memory access                                                            Yes      Yes
:cpp:func:`LeaveDataPtrCOO <rocalution::LocalMatrix::LeaveDataPtrCOO>`               Direct Memory access                                                            Yes      Yes
:cpp:func:`LeaveDataPtrDIA <rocalution::LocalMatrix::LeaveDataPtrDIA>`               Direct Memory access                                                            Yes      Yes
:cpp:func:`LeaveDataPtrELL <rocalution::LocalMatrix::LeaveDataPtrELL>`               Direct Memory access                                                            Yes      Yes
:cpp:func:`LeaveDataPtrDENSE <rocalution::LocalMatrix::LeaveDataPtrDENSE>`           Direct Memory access                                                            Yes      Yes
:cpp:func:`Zeros <rocalution::LocalMatrix::Zeros>`                                   Set all matrix entries to zero                                                  Yes      Yes
:cpp:func:`Scale <rocalution::LocalMatrix::Scale>`                                   Scale all matrix non-zeros                                                      Yes      Yes
:cpp:func:`ScaleDiagonal <rocalution::LocalMatrix::ScaleDiagonal>`                   Scale matrix diagonal                                                           Yes      Yes
:cpp:func:`ScaleOffDiagonal <rocalution::LocalMatrix::ScaleOffDiagonal>`             Scale matrix off-diagonal entries                                               Yes      Yes
:cpp:func:`AddScalar <rocalution::LocalMatrix::AddScalar>`                           Add scalar to all matrix non-zeros                                              Yes      Yes
:cpp:func:`AddScalarDiagonal <rocalution::LocalMatrix::AddScalarDiagonal>`           Add scalar to matrix diagonal                                                   Yes      Yes
:cpp:func:`AddScalarOffDiagonal <rocalution::LocalMatrix::AddScalarOffDiagonal>`     Add scalar to matrix off-diagonal entries                                       Yes      Yes
:cpp:func:`ExtractSubMatrix <rocalution::LocalMatrix::ExtractSubMatrix>`             Extract sub-matrix                                                              Yes      Yes
:cpp:func:`ExtractSubMatrices <rocalution::LocalMatrix::ExtractSubMatrices>`         Extract array of non-overlapping sub-matrices                                   Yes      Yes
:cpp:func:`ExtractDiagonal <rocalution::LocalMatrix::ExtractDiagonal>`               Extract matrix diagonal                                                         Yes      Yes
:cpp:func:`ExtractInverseDiagonal <rocalution::LocalMatrix::ExtractInverseDiagonal>` Extract inverse matrix diagonal                                                 Yes      Yes
:cpp:func:`ExtractL <rocalution::LocalMatrix::ExtractL>`                             Extract lower triangular matrix                                                 Yes      Yes
:cpp:func:`ExtractU <rocalution::LocalMatrix::ExtractU>`                             Extract upper triangular matrix                                                 Yes      Yes
:cpp:func:`Permute <rocalution::LocalMatrix::Permute>`                               (Forward) permute the matrix                                                    Yes      Yes
:cpp:func:`PermuteBackward <rocalution::LocalMatrix::PermuteBackward>`               (Backward) permute the matrix                                                   Yes      Yes
:cpp:func:`CMK <rocalution::LocalMatrix::CMK>`                                       Create CMK permutation vector                                                   Yes      No
:cpp:func:`RCMK <rocalution::LocalMatrix::RCMK>`                                     Create reverse CMK permutation vector                                           Yes      No
:cpp:func:`ConnectivityOrder <rocalution::LocalMatrix::ConnectivityOrder>`           Create connectivity (increasing nnz per row) permutation vector                 Yes      No
:cpp:func:`MultiColoring <rocalution::LocalMatrix::MultiColoring>`                   Create multi-coloring decomposition of the matrix                               Yes      No
:cpp:func:`MaximalIndependentSet <rocalution::LocalMatrix::MaximalIndependentSet>`   Create maximal independent set decomposition of the matrix                      Yes      No
:cpp:func:`ZeroBlockPermutation <rocalution::LocalMatrix::ZeroBlockPermutation>`     Create permutation where zero diagonal entries are mapped to the last block     Yes      No
:cpp:func:`ILU0Factorize <rocalution::LocalMatrix::ILU0Factorize>`                   Create ILU(0) factorization                                                     Yes      No
:cpp:func:`LUFactorize <rocalution::LocalMatrix::LUFactorize>`                       Create LU factorization                                                         Yes      No
:cpp:func:`ILUTFactorize <rocalution::LocalMatrix::ILUTFactorize>`                   Create ILU(t,m) factorization                                                   Yes      No
:cpp:func:`ILUpFactorize <rocalution::LocalMatrix::ILUpFactorize>`                   Create ILU(p) factorization                                                     Yes      No
:cpp:func:`ICFactorize <rocalution::LocalMatrix::ICFactorize>`                       Create IC factorization                                                         Yes      No
:cpp:func:`QRDecompose <rocalution::LocalMatrix::QRDecompose>`                       Create QR decomposition                                                         Yes      No
:cpp:func:`ReadFileMTX <rocalution::LocalMatrix::ReadFileMTX>`                       Read matrix from matrix market file                                             Yes      No
:cpp:func:`WriteFileMTX <rocalution::LocalMatrix::WriteFileMTX>`                     Write matrix to matrix market file                                              Yes      No
:cpp:func:`ReadFileCSR <rocalution::LocalMatrix::ReadFileCSR>`                       Read matrix from binary file                                                    Yes      No
:cpp:func:`WriteFileCSR <rocalution::LocalMatrix::WriteFileCSR>`                     Write matrix to binary file                                                     Yes      No
:cpp:func:`CopyFrom <rocalution::LocalMatrix::CopyFrom>`                             Copy matrix (values and structure) from another LocalMatrix                     Yes      Yes
:cpp:func:`CopyFromAsync <rocalution::LocalMatrix::CopyFromAsync>`                   Copy matrix asynchronously                                                      Yes      Yes
:cpp:func:`CloneFrom <rocalution::LocalMatrix::CloneFrom>`                           Clone an entire matrix (values, structure and backend) from another LocalMatrix Yes      Yes
:cpp:func:`UpdateValuesCSR <rocalution::LocalMatrix::UpdateValuesCSR>`               Update CSR matrix values (structure remains identical)                          Yes      Yes
:cpp:func:`CopyFromCSR <rocalution::LocalMatrix::CopyFromCSR>`                       Copy (import) CSR matrix                                                        Yes      Yes
:cpp:func:`CopyToCSR <rocalution::LocalMatrix::CopyToCSR>`                           Copy (export) CSR matrix                                                        Yes      Yes
:cpp:func:`CopyFromCOO <rocalution::LocalMatrix::CopyFromCOO>`                       Copy (import) COO matrix                                                        Yes      Yes
:cpp:func:`CopyToCOO <rocalution::LocalMatrix::CopyToCOO>`                           Copy (export) COO matrix                                                        Yes      Yes
:cpp:func:`CopyFromHostCSR <rocalution::LocalMatrix::CopyFromHostCSR>`               Allocate and copy (import) a CSR matrix from host                               Yes      No
:cpp:func:`ConvertToCSR <rocalution::LocalMatrix::ConvertToCSR>`                     Convert a matrix to CSR format                                                  Yes      No
:cpp:func:`ConvertToMCSR <rocalution::LocalMatrix::ConvertToMCSR>`                   Convert a matrix to MCSR format                                                 Yes      No
:cpp:func:`ConvertToBCSR <rocalution::LocalMatrix::ConvertToBCSR>`                   Convert a matrix to BCSR format                                                 Yes      No
:cpp:func:`ConvertToCOO <rocalution::LocalMatrix::ConvertToCOO>`                     Convert a matrix to COO format                                                  Yes      Yes
:cpp:func:`ConvertToELL <rocalution::LocalMatrix::ConvertToELL>`                     Convert a matrix to ELL format                                                  Yes      Yes
:cpp:func:`ConvertToDIA <rocalution::LocalMatrix::ConvertToDIA>`                     Convert a matrix to DIA format                                                  Yes      Yes
:cpp:func:`ConvertToHYB <rocalution::LocalMatrix::ConvertToHYB>`                     Convert a matrix to HYB format                                                  Yes      Yes
:cpp:func:`ConvertToDENSE <rocalution::LocalMatrix::ConvertToDENSE>`                 Convert a matrix to DENSE format                                                Yes      No
:cpp:func:`ConvertTo <rocalution::LocalMatrix::ConvertTo>`                           Convert a matrix                                                                Yes
:cpp:func:`SymbolicPower <rocalution::LocalMatrix::SymbolicPower>`                   Perform symbolic power computation (structure only)                             Yes      No
:cpp:func:`MatrixAdd <rocalution::LocalMatrix::MatrixAdd>`                           Matrix addition                                                                 Yes      No
:cpp:func:`MatrixMult <rocalution::LocalMatrix::MatrixMult>`                         Multiply two matrices                                                           Yes      No
:cpp:func:`DiagonalMatrixMult <rocalution::LocalMatrix::DiagonalMatrixMult>`         Multiply matrix with diagonal matrix (stored in LocalVector)                    Yes      Yes
:cpp:func:`DiagonalMatrixMultL <rocalution::LocalMatrix::DiagonalMatrixMultL>`       Multiply matrix with diagonal matrix (stored in LocalVector) from left          Yes      Yes
:cpp:func:`DiagonalMatrixMultR <rocalution::LocalMatrix::DiagonalMatrixMultR>`       Multiply matrix with diagonal matrix (stored in LocalVector) from right         Yes      Yes
:cpp:func:`Gershgorin <rocalution::LocalMatrix::Gershgorin>`                         Compute the spectrum approximation with Gershgorin circles theorem              Yes      No
:cpp:func:`Compess <rocalution::LocalMatrix::Compress>`                              Delete all entries where `abs(a_ij) <= drop_off`                                Yes      Yes
:cpp:func:`Transpose <rocalution::LocalMatrix::Transpose>`                           Transpose the matrix                                                            Yes      No
:cpp:func:`Sort <rocalution::LocalMatrix::Sort>`                                     Sort the matrix indices                                                         Yes      No
:cpp:func:`Key <rocalution::LocalMatrix::Key>`                                       Compute a unique matrix key                                                     Yes      No
:cpp:func:`ReplaceColumnVector <rocalution::LocalMatrix::ReplaceColumnVector>`       Replace a column vector of a matrix                                             Yes      No
:cpp:func:`ReplaceRowVector <rocalution::LocalMatrix::ReplaceRowVector>`             Replace a row vector of a matrix                                                Yes      No
:cpp:func:`ExtractColumnVector <rocalution::LocalMatrix::ExtractColumnVector>`       Extract a column vector of a matrix                                             Yes      No
:cpp:func:`ExtractRowVector <rocalution::LocalMatrix::ExtractRowVector>`             Extract a row vector of a matrix                                                Yes      No
==================================================================================== =============================================================================== ======== =======

====================================================================================== ===================================================================== ======== =======
**LocalVector function**                                                               **Comment**                                                           **Host** **HIP**
====================================================================================== ===================================================================== ======== =======
:cpp:func:`GetSize <rocalution::LocalVector::GetSize>`                                 Obtain vector size                                                    Yes      Yes
:cpp:func:`Check <rocalution::LocalVector::Check>`                                     Check vector for valid entries                                        Yes      No
:cpp:func:`Allocate <rocalution::LocalVector::Allocate>`                               Allocate vector                                                       Yes      Yes
:cpp:func:`Sync <rocalution::LocalVector::Sync>`                                       Synchronize                                                           Yes      Yes
:cpp:func:`SetDataPtr <rocalution::LocalVector::SetDataPtr>`                           Initialize vector with external data                                  Yes      Yes
:cpp:func:`LeaveDataPtr <rocalution::LocalVector::LeaveDataPtr>`                       Direct Memory Access                                                  Yes      Yes
:cpp:func:`Zeros <rocalution::LocalVector::Zeros>`                                     Set vector entries to zero                                            Yes      Yes
:cpp:func:`Ones <rocalution::LocalVector::Ones>`                                       Set vector entries to one                                             Yes      Yes
:cpp:func:`SetValues <rocalution::LocalVector::SetValues>`                             Set vector entries to scalar                                          Yes      Yes
:cpp:func:`SetRandomUniform <rocalution::LocalVector::SetRandomUniform>`               Initialize vector with uniformly distributed random numbers           Yes      No
:cpp:func:`SetRandomNormal <rocalution::LocalVector::SetRandomNorm>`                   Initialize vector with normally distributed random numbers            Yes      No
:cpp:func:`ReadFileASCII <rocalution::LocalVector::ReadFileASCII>`                     Read vector for ASCII file                                            Yes      No
:cpp:func:`WriteFileASCII <rocalution::LocalVector::WriteFileASCII>`                   Write vector to ASCII file                                            Yes      No
:cpp:func:`ReadFileBinary <rocalution::LocalVector::ReadFileBinary>`                   Read vector from binary file                                          Yes      No
:cpp:func:`WriteFileBinary <rocalution::LocalVector::WriteFileBinary>`                 Write vector to binary file                                           Yes      No
:cpp:func:`CopyFrom <rocalution::LocalVector::CopyFrom>`                               Copy vector (values) from another LocalVector                         Yes      Yes
:cpp:func:`CopyFromAsync <rocalution::LocalVector::CopyFromAsync>`                     Copy vector asynchronously                                            Yes      Yes
:cpp:func:`CopyFromFloat <rocalution::LocalVector::CopyFromFloat>`                     Copy vector from another LocalVector<float>                           Yes      Yes
:cpp:func:`CopyFromDouble <rocalution::LocalVector::CopyFromDouble>`                   Copy vector from another LocalVector<double>                          Yes      Yes
:cpp:func:`CopyFromPermute <rocalution::LocalVector::CopyFromPermute>`                 Copy vector under specified (forward) permutation                     Yes      Yes
:cpp:func:`CopyFromPermuteBackward <rocalution::LocalVector::CopyFromPermuteBackward>` Copy vector under specified (backward) permutation                    Yes      Yes
:cpp:func:`CloneFrom <rocalution::LocalVector::CloneFrom>`                             Clone vector (values and backend descriptor) from another LocalVector Yes      Yes
:cpp:func:`CopyFromData <rocalution::LocalVector::CopyFromData>`                       Copy (import) vector from array                                       Yes      Yes
:cpp:func:`CopyToData <rocalution::LocalVector::CopyToData>`                           Copy (export) vector to array                                         Yes      Yes
:cpp:func:`Permute <rocalution::LocalVector::Permute>`                                 (Foward) permute vector in-place                                      Yes      Yes
:cpp:func:`PermuteBackward <rocalution::LocalVector::PermuteBackward>`                 (Backward) permute vector in-place                                    Yes      Yes
:cpp:func:`AddScale <rocalution::LocalVector::AddScale>`                               `y = a * x + y`                                                       Yes      Yes
:cpp:func:`ScaleAdd <rocalution::LocalVector::ScaleAdd>`                               `y = x + a * y`                                                       Yes      Yes
:cpp:func:`ScaleAddScale <rocalution::LocalVector::ScaleAddScale>`                     `y = b * x + a * y`                                                   Yes      Yes
:cpp:func:`ScaleAdd2 <rocalution::LocalVector::ScaleAdd2>`                             `z = a * x + b * y + c * z`                                           Yes      Yes
:cpp:func:`Scale <rocalution::LocalVector::Scale>`                                     `x = a * x`                                                           Yes      Yes
:cpp:func:`ExclusiveScan <rocalution::LocalVector::ExclusiveScan>`                     Compute exclusive sum                                                 Yes      No
:cpp:func:`Dot <rocalution::LocalVector::Dot>`                                         Compute dot product                                                   Yes      Yes
:cpp:func:`DotNonConj <rocalution::LocalVector::DotNonConj>`                           Compute non-conjugated dot product                                    Yes      Yes
:cpp:func:`Norm <rocalution::LocalVector::Norm>`                                       Compute L2 norm                                                       Yes      Yes
:cpp:func:`Reduce <rocalution::LocalVector::Reduce>`                                   Obtain the sum of all vector entries                                  Yes      Yes
:cpp:func:`Asum <rocalution::LocalVector::Asum>`                                       Obtain the absolute sum of all vector entries                         Yes      Yes
:cpp:func:`Amax <rocalution::LocalVector::Amax>`                                       Obtain the absolute maximum entry of the vector                       Yes      Yes
:cpp:func:`PointWiseMult <rocalution::LocalVector::PointWiseMult>`                     Perform point wise multiplication of two vectors                      Yes      Yes
:cpp:func:`Power <rocalution::LocalVector::Power>`                                     Compute vector power                                                  Yes      Yes
====================================================================================== ===================================================================== ======== =======

Solver and preconditioner classes
=================================

.. note:: 

   The building phase of the iterative solver also depends on the selected preconditioner.

================================================================= ================= ======== =======
**Solver**                                                        **Functionality** **Host** **HIP**
================================================================= ================= ======== =======
:cpp:class:`CG <rocalution::CG>`                                  Building          Yes      Yes
:cpp:class:`CG <rocalution::CG>`                                  Solving           Yes      Yes
:cpp:class:`FCG <rocalution::FCG>`                                Building          Yes      Yes
:cpp:class:`FCG <rocalution::FCG>`                                Solving           Yes      Yes
:cpp:class:`CR <rocalution::CR>`                                  Building          Yes      Yes
:cpp:class:`CR <rocalution::CR>`                                  Solving           Yes      Yes
:cpp:class:`BiCGStab <rocalution::BiCGStab>`                      Building          Yes      Yes
:cpp:class:`BiCGStab <rocalution::BiCGStab>`                      Solving           Yes      Yes
:cpp:class:`BiCGStab(l) <rocalution::BiCGStabl>`                  Building          Yes      Yes
:cpp:class:`BiCGStab(l) <rocalution::BiCGStabl>`                  Solving           Yes      Yes
:cpp:class:`QMRCGStab <rocalution::QMRCGStab>`                    Building          Yes      Yes
:cpp:class:`QMRCGStab <rocalution::QMRCGStab>`                    Solving           Yes      Yes
:cpp:class:`GMRES <rocalution::GMRES>`                            Building          Yes      Yes
:cpp:class:`GMRES <rocalution::GMRES>`                            Solving           Yes      Yes
:cpp:class:`FGMRES <rocalution::FGMRES>`                          Building          Yes      Yes
:cpp:class:`FGMRES <rocalution::FGMRES>`                          Solving           Yes      Yes
:cpp:class:`Chebyshev <rocalution::Chebyshev>`                    Building          Yes      Yes
:cpp:class:`Chebyshev <rocalution::Chebyshev>`                    Solving           Yes      Yes
:cpp:class:`Mixed-Precision <rocalution::MixedPrecisionDC>`       Building          Yes      Yes
:cpp:class:`Mixed-Precision <rocalution::MixedPrecisionDC>`       Solving           Yes      Yes
:cpp:class:`Fixed-Point Iteration <rocalution::FixedPoint>`       Building          Yes      Yes
:cpp:class:`Fixed-Point Iteration <rocalution::FixedPoint>`       Solving           Yes      Yes
:cpp:class:`AMG (Plain Aggregation) <rocalution::UAAMG>`          Building          Yes      No
:cpp:class:`AMG (Plain Aggregation) <rocalution::UAAMG>`          Solving           Yes      Yes
:cpp:class:`AMG (Smoothed Aggregation) <rocalution::SAAMG>`       Building          Yes      No
:cpp:class:`AMG (Smoothed Aggregation) <rocalution::SAAMG>`       Solving           Yes      Yes
:cpp:class:`AMG (Ruge Stueben) <rocalution::RugeStuebenAMG>`      Building          Yes      No
:cpp:class:`AMG (Ruge Stueben) <rocalution::RugeStuebenAMG>`      Solving           Yes      Yes
:cpp:class:`AMG (Pairwise Aggregation) <rocalution::PairwiseAMG>` Building          Yes      No
:cpp:class:`AMG (Pairwise Aggregation) <rocalution::PairwiseAMG>` Solving           Yes      Yes
:cpp:class:`LU <rocalution::LU>`                                  Building          Yes      No
:cpp:class:`LU <rocalution::LU>`                                  Solving           Yes      No
:cpp:class:`QR <rocalution::QR>`                                  Building          Yes      No
:cpp:class:`QR <rocalution::QR>`                                  Solving           Yes      No
:cpp:class:`Inversion <rocalution::Inversion>`                    Building          Yes      No
:cpp:class:`Inversion <rocalution::Inversion>`                    Solving           Yes      Yes
================================================================= ================= ======== =======

=================================================================== ================= ======== =======
**Preconditioner**                                                  **Functionality** **Host** **HIP**
=================================================================== ================= ======== =======
:cpp:class:`Jacobi <rocalution::Jacobi>`                            Building          Yes      Yes
:cpp:class:`Jacobi <rocalution::Jacobi>`                            Solving           Yes      Yes
:cpp:class:`BlockJacobi <rocalution::BlockJacobi>`                  Building          Yes      Yes
:cpp:class:`BlockJacobi <rocalution::BlockJacobi>`                  Solving           Yes      Yes
:cpp:class:`MultiColoredILU(0,1) <rocalution::MultiColoredILU>`     Building          Yes      Yes
:cpp:class:`MultiColoredILU(0,1) <rocalution::MultiColoredILU>`     Solving           Yes      Yes
:cpp:class:`MultiColoredILU(>0, >1) <rocalution::MultiColoredILU>`  Building          Yes      No
:cpp:class:`MultiColoredILU(>0, >1) <rocalution::MultiColoredILU>`  Solving           Yes      Yes
:cpp:class:`MultiElimination(I)LU <rocalution::MultiElimination>`   Building          Yes      No
:cpp:class:`MultiElimination(I)LU <rocalution::MultiElimination>`   Solving           Yes      Yes
:cpp:class:`ILU(0) <rocalution::ILU>`                               Building          Yes      Yes
:cpp:class:`ILU(0) <rocalution::ILU>`                               Solving           Yes      Yes
:cpp:class:`ILU(>0) <rocalution::ILU>`                              Building          Yes      No
:cpp:class:`ILU(>0) <rocalution::ILU>`                              Solving           Yes      No
:cpp:class:`ILUT <rocalution::ILUT>`                                Building          Yes      No
:cpp:class:`ILUT <rocalution::ILUT>`                                Solving           Yes      No
:cpp:class:`IC(0) <rocalution::IC>`                                 Building          Yes      No
:cpp:class:`IC(0) <rocalution::IC>`                                 Solving           Yes      No
:cpp:class:`FSAI <rocalution::FSAI>`                                Building          Yes      No
:cpp:class:`FSAI <rocalution::FSAI>`                                Solving           Yes      Yes
:cpp:class:`SPAI <rocalution::SPAI>`                                Building          Yes      No
:cpp:class:`SPAI <rocalution::SPAI>`                                Solving           Yes      Yes
:cpp:class:`Chebyshev <rocalution::AIChebyshev>`                    Building          Yes      No
:cpp:class:`Chebyshev <rocalution::AIChebyshev>`                    Solving           Yes      Yes
:cpp:class:`MultiColored(S)GS <rocalution::MultiColoredSGS>`        Building          Yes      No
:cpp:class:`MultiColored(S)GS <rocalution::MultiColoredSGS>`        Solving           Yes      Yes
:cpp:class:`(S)GS <rocalution::SGS>`                                Building          Yes      No
:cpp:class:`(S)GS <rocalution::SGS>`                                Solving           Yes      No
:cpp:class:`(R)AS <rocalution::AS>`                                 Building          Yes      Yes
:cpp:class:`(R)AS <rocalution::AS>`                                 Solving           Yes      Yes
:cpp:class:`BlockPreconditioner <rocalution::BlockPreconditioner>`  Building          Yes      Yes
:cpp:class:`BlockPreconditioner <rocalution::BlockPreconditioner>`  Solving           Yes      Yes
:cpp:class:`SaddlePoint <rocalution::DiagJacobiSaddlePointPrecond>` Building          Yes      No
:cpp:class:`SaddlePoint <rocalution::DiagJacobiSaddlePointPrecond>` Solving           Yes      Yes
=================================================================== ================= ======== =======
