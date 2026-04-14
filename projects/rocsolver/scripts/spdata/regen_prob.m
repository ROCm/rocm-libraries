% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function [T,Q] = regen_prob( dir )
%  [T,Q] = regen_prob( dir )
%
%  Regenerate problem by reading in the original sparse matrix
%  then regenerating and writing out the permutations and
%  factors
%
A = read_csr( 'ptrA','indA','valA', dir);
[T,Q] = print_prob( A, dir );

end



