% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function A=csr2full(m,ptr,ind,val)

%{
-----------------------------------------------------------
  	This function takes a matrix in CSR format and returns a corresponding
  	full matrix with m rows.

  	A = csr2full(m, ptr, ind, val)

  	Inputs:
  	m  					The number of rows of the output matrix
  	ptr, ind, val 		The CSR format arrays

  	Outputs:
  	A  					The output matrix (a general 2D array)

  	(This script is for internal use only. It is not part of
	rocSOLVER library interface and could change or be removed
	without any notice)
-----------------------------------------------------------
%}


A=zeros(m,m);

for i = 1:m,
	k = ptr(i+1) - ptr(i);
	for j=1:k,
		A(i, ind(ptr(i)+j) + 1) = val(ptr(i)+j);
	end;
end;

end

