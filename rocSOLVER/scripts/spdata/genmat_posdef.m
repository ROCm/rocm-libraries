% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function genmat_posdef(n,nz)

%{
-----------------------------------------------------------
	This function generates all the files required for a
	test case of the Cholesky re-factorization functionality in rocSOLVER.

	genmat_gen(n, nz)

	Inputs:
	n  			Size of the problem (number of row and columns
				of the sparse positive-definite matrix A)
	nz  		Max number of non-zero elements in A

	Outputs:
	isok		This will be -1 if any of the files could not
				be generated

	The resulting files are created in the working directory.
	See printall.m for more details.

	(This script is for internal use only. It is not part of
	rocSOLVER library interface and could change or be removed
	without any notice)
-----------------------------------------------------------
%}


%---------------------------
% generate random sparse matrix
% with nz non-zero elements
%---------------------------
per = nz / (n*n);
A = sprand(n, n, per);

%---------------------------
% Take lower part L and make it diagonally dominant
% (this means that the symmetric matrix L+L^T will be
% positive definite)
%---------------------------
A=tril(A);
A = fixmat(A, n/2);

%---------------------------
% If necessary, remove extra non-zero elements
% so that the final symmetric matrix has no more
% than nz non-zero elements
%---------------------------
nzsym = (nz + n) / 2;
dif = nnz(A) - nzsym;
if(dif > 0),
	A = cleanmat(A, dif);
end;

%---------------------------
% compute the final symmetric matrix
% (functions work with only the lower part, but better
% to have a full symmetric to generate the linear systems)
%---------------------------
A = A + transpose(A);

%---------------------------
% Compute Cholesky factorization
%---------------------------
[T, p, Q] = chol(A, 'vector', 'lower');

%---------------------------
% Generate output files
%---------------------------
disp('')
disp('Generating files...')
isok = printall("po", A, T, Q);
if (isok),
	disp('OK')
	disp('')
else
	disp('FAILURE!');
	disp('')
end;

%------------------------
% quick validation
%------------------------
ptr = dlmread('ptrT');
ind = dlmread('indT');
val = dlmread('valT');
t = csr2full(n, ptr, ind, val);

qq = dlmread('Q') + 1;
q = eye(n,n);
q = q(:,qq);

disp('Quick checking:')
b = dlmread('B_1');
x = dlmread('X_1');
err1 = norm(q*t*t'*q'*x-b, "fro")
b = dlmread('B_10');
x = dlmread('X_10');
err2 = norm(q*t*t'*q'*x-b, "fro")
b = dlmread('B_30');
x = dlmread('X_30');
err3 = norm(q*t*t'*q'*x-b, "fro")
disp('')

end
