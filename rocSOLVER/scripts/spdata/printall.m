% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function isok = printall(modo,A,T,Q,L=0,U=0,P=0)

%{
-----------------------------------------------------------
	This function prints into files all the involved
	matrices and vectors of the LU factorization

	P * A * Q = L * U,   T = L - I + U

	if modo is "lu", or the Cholesky factorization

	Q' * A * Q = T * T'

	otherwise.

	Additionaly, it generates and prints matrix B and
	solution matrix X of the linear systems

	A * X = B

	for 1, 10 and 30 right-hand-sides.

	printall("lu", A, T, Q, L, U, P)
	printall("po", A, T, Q)

	Inputs:
	A, L, U, T 		The matrix A and its corresponding
					triangular factors (sparse matrices)
	P, Q			vectors with the indices corresponding
					to the row and column re-orderings (these could be
					row or column full vectors)

	Outputs:
	isok			This will be -1 if any of the matrix/vectors could not
					be printed

	The resulting files are created in the working directory.
	See print_mat.m, print_x.m and print_vec.m for more details.

	(This script is for internal use only. It is not part of
	rocSOLVER library interface and could change or be removed
	without any notice)
-----------------------------------------------------------
%}


isok = print_mat('A',A);
if (!isok),
	error(sprintf('Error writting files for matrix A'));
end;

isok = print_mat('T',T);
if (!isok),
  	error(sprintf('Error writting files for matrix T'));
end;

isok = print_vec('Q',Q);
if (!isok),
  	error(sprintf('Error writting files for vector Q'));
end;

if modo=="lu",
	isok = print_mat('L',L);
	if (!isok),
  		error(sprintf('Error writting files for matrix L'));
	end;

	isok = print_mat('U',U);
	if (!isok),
  		error(sprintf('Error writting files for matrix U'));
	end;

	isok = print_vec('P',P);
	if (!isok),
  		error(sprintf('Error writting files for vector P'));
	end;
end;

n=size(A,1);

X=rand(n,1);
B=A*X;
isok = print_x('X_1',X);
if (!isok),
  	error(sprintf('Error writting files for matrix X with 1 rhs'));
end;
isok = print_x('B_1',B);
if (!isok),
  	error(sprintf('Error writting files for matrix B with 1 rhs'));
end;

X=rand(n,10);
B=A*X;
isok = print_x('X_10',X);
if (!isok),
  	error(sprintf('Error writting files for matrix X with 10 rhs'));
end;
isok = print_x('B_10',B);
if (!isok),
  	error(sprintf('Error writting files for matrix B with 10 rhs'));
end;

X=rand(n,30);
B=A*X;
isok = print_x('X_30',X);
if (!isok),
  	error(sprintf('Error writting files for matrix X with 30 rhs'));
end;
isok = print_x('B_30',B);
if (!isok),
  	error(sprintf('Error writting files for matrix B with 30 rhs'));
end;

end
