% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************
function A = spdiag( d, idiag )
% A = spdiag( d, idiag )
%
% Generate a sparse matrix based on the diagonal
% This is similar to "diag" but returns
% a sparse matrix instead of a dense matrix

n = length(d);
m = n + abs(idiag);

if (idiag >= 0),
   % ----------------------
   % upper triangular part
   % ----------------------
   i1 = 1;
   i2 = min(m,i1 + n - 1);
   j1 = min(m,1 + idiag);
   j2 = min(m,j1 + n -1 );
else
   % -----------------------
   % lower triangular part
   % -----------------------
   i1 = min(m,1 + abs(idiag));
   i2 = min(m, i1 + n - 1);
   j1 = 1;
   j2 = min(m, j1 + n - 1);
end;


A = sparse( i1:i2, j1:j2, d, m,m );

