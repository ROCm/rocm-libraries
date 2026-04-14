% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function genmat_gen(n,nz)

%{
-----------------------------------------------------------
	This function generates all the files required for a
	test case of the LU re-factorization functionality in rocSOLVER.

	genmat_gen(n, nz)

	Inputs:
	n  			Size of the problem (number of row and columns
				of the sparse matrix A)
	nz  		Number of non-zero elements in A

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
% Ensure matrix is diagonally dominant
%---------------------------
A = fixmat(A, n);

%---------------------------
% If necessary, remove extra non-zero elements
%---------------------------
dif = nnz(A) - nz;
if(dif > 0),
	A = cleanmat(A, dif);
end;

%---------------------------
% Compute LU factorization
%---------------------------
[L, U, P, Q] = lu(A, 'vector');
T = L - eye(n,n) + U;

%---------------------------
% Generate output files
%---------------------------
disp('')
disp('Generating files...')
isok = printall("lu", A, T, Q, L, U, P);
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
l = tril(t) - diag(diag(t)) + eye(n,n);
u = triu(t);

pp = dlmread('P') + 1;
p = eye(n,n);
p = p(pp,:);
qq = dlmread('Q') + 1;
q = eye(n,n);
q = q(:,qq);

disp('Quick checking:')
b = dlmread('B_1');
x = dlmread('X_1');
err1 = norm(p'*l*u*q'*x-b, "fro")
b = dlmread('B_10');
x = dlmread('X_10');
err2 = norm(p'*l*u*q'*x-b, "fro")
b = dlmread('B_30');
x = dlmread('X_30');
err3 = norm(p'*l*u*q'*x-b, "fro")
disp('')


end
