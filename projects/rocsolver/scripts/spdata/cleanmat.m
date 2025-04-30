% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function A=cleanmat(B,toclean)

%{
-------------------------------------------------------------
	This function makes zero 'toclean' non-zero
	elements of B that are not in the diagonal.

	A = cleanmat(B, toclean)

	inputs:
	B 			the original matrix (sparse matrix)
	toclean		the number of elements to be removed

	outputs:
	A  			the modified resulting matrix (sparse matrix)

	(This script is for internal use only. It is not part of
	rocSOLVER library interface and could change or be removed
	without any notice)
-------------------------------------------------------------
%}


n = size(B,1);
A=full(B);

c = 0;
while c < toclean,
	for i=1:n,
		cc = 0;
		for j=1:n,
			if (A(i,j) != 0 && i != j),
				A(i,j) = 0;
				cc = 1;
				break;
			end;
		end;
		if cc == 1,
			c = c + 1;
			break;
		end;
	end;
end;

A=sparse(A);

end
