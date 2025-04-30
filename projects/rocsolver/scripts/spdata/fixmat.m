% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function A=fixmat(B,diagval)

%{
-------------------------------------------------------------
	This function sets all diagonal entries of B to 'diagval'
	without changing the number of nonzero entries in the
	corresponding columns of the matrix.

	A = fixmat(B, diagval)

	inputs:
	B 			the original matrix (sparse matrix)
	diagval		the value for the diagonal entries

	outputs:
	A  			the modified resulting matrix (sparse matrix)

	(This script is for internal use only. It is not part of
	rocSOLVER library interface and could change or be removed
	without any notice)
-------------------------------------------------------------
%}


n = size(B,1);
A=full(B);

for j=1:n,
	if (A(j,j) == 0),
		for i=1:n,
			if (A(i,j) != 0),
				A(i,j) = 0;
				break;
			end;
		end;
	end;
	A(j,j)=diagval;
end;

A=sparse(A);

end




