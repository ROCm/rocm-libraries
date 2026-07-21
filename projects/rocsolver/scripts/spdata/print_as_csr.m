% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************
`
function isok = print_as_csr( A, name )
%
% isok = print_as_csr( A, name )
%
% print sparse matrix A as
% Compressed Sparse Row (CSR) sparse matrix storage format
% with the pointers, column indices and numerical values split into 3 files
%
% For example,
% isok = print_as_csr(T,'T')
%
% will generate files 'ptrT', 'indT', 'valT'
%
% isok will be 0 (false) if there are errors

[Ap,Ai,Ax] = sparse2csr(A);
isok_Ap = print_gvec( sprintf('ptr%s',name), Ap );
isok_Ai = print_gvec( sprintf('ind%s',name), Ai );
isok_Ax = print_gvec( sprintf('val%s',name), Ax );
isok = isok_Ap & isok_Ai & isok_Ax;

end
