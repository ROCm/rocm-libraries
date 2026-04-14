% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function [ptr,ind,val] = full2csr(A)

%{
-----------------------------------------------------------
  	This function takes a square matrix A and generates arrays
  	ptr, ind and val corresponding to its Compressed Row Storage (CSR) format.

  	[ptr, ind, val] = full2csr(A)

  	Inputs:
  	A  					The square matrix (a general 2D array)

  	Outputs:
  	ptr, ind, val 		The corresponding CSR format arrays

  	(This script is for internal use only. It is not part of
	rocSOLVER library interface and could change or be removed
	without any notice)
-----------------------------------------------------------
%}

n = size(A,1);

ptr = 0;
ind = [];
val = [];

for i=1:n,
	c=0;
	for j=1:n,
		if A(i,j) != 0
			c = c+1;
			ind = [ind j-1];
			val = [val A(i,j)];
		end;
	end;
	ptr(i+1) = ptr(i) + c;
end;

end
