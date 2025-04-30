% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function [T,Q] = print_prob( A, dir_in )
%
% [T,Q] = print_prob( A [, dir] )
%
%
% Given a sparse matrix, write out the permutations and factorization
%
% The files are created in the current local directory by default
% If optional "dir" is given, the files are created under that directory

dir = '.';
if (nargin >= 2),
  dir = dir_in;
  is_absolute_dir = (dir(1) == '/');
  if (is_absolute_dir),
    [status,msg,msgid] = mkdir('',dir);
  else
    [status,msg,msgid] = mkdir('.',dir);
  end;
  isok = (status == 1);
  if (~isok),
    error(sprintf('print_prob: mkdir for %s return msg=%s, msgid=%g', ...
                                         dir,      msg,    msgid ));
    return;
  end;
end;

has_final_slash = (dir( numel(dir) ) == '/');
if (~has_final_slash),
  dir = strcat(dir,'/');
end;

nrow = size(A,1);
ncol = size(A,2);
[Ap,Ai,Ax] = sparse2csr( A );
print_gvec(strcat(dir,'ptrA'),Ap);
print_gvec(strcat(dir,'indA'),Ai);
print_gvec(strcat(dir,'valA'),Ax);



m = 30;
X = reshape( 1:(ncol*m), ncol, m );
B = A * X;

print_gvec(strcat(dir,'X_30'),X(:,1:30));
print_gvec(strcat(dir,'X_10'),X(:,1:10));
print_gvec(strcat(dir,'X_1'),X(:,1:1));

print_gvec(strcat(dir,'B_30'),B(:,1:30));
print_gvec(strcat(dir,'B_10'),B(:,1:10));
print_gvec(strcat(dir,'B_1'),B(:,1:1));


[L, is_indefinite, Q] = chol(A,'lower');
if (is_indefinite),

% perform LU instead

[L,U,P,Q] = lu(A);

np = size(P,2);
pvec = P*reshape(1:np,np,1);
pvec = pvec - 1;
print_gvec(strcat(dir,'P'),pvec);

nq = size(Q,2);
qvec = Q * reshape(1:nq,nq,1);
qvec = qvec - 1;
print_gvec(strcat(dir,'Q'),qvec);


T = tril(L,-1) + triu(U);


else


tiny = 1e-100;
T = L + triu(L',1)*tiny + triu(Q'*A*Q,1);

nq = size(Q,2);
qvec = Q * reshape(1:nq,nq,1);
qvec = qvec - 1;
print_gvec(strcat(dir,'Q'),qvec);

end



[Tp,Ti,Tx] = sparse2csr( T );




print_gvec(strcat(dir,'ptrT'),Tp);
print_gvec(strcat(dir,'indT'),Ti);
print_gvec(strcat(dir,'valT'),Tx);



end
