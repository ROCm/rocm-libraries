% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function isok = print_mat(matname,A)

%{
-----------------------------------------------------------
  This function prints into files ptr, ind and val the
  arrays corresponding to the sparse matrix A.

  isok = print_mat(matname, A)

  Inputs:
  matname     The common suffix added to the generated files names
  A           Sparse matrix to be printed

  Outputs:
  isok        This will be -1 if any of the file operations failed

  The resulting files

    ptr{matname}
    ind{matname}
    val{matname}

  are created in the working directory.

  (This script is for internal use only. It is not part of
  rocSOLVER library interface and could change or be removed
  without any notice)
-----------------------------------------------------------
%}


% -----------------------
% open files
% -----------------------

[ptr,msg] = fopen(strcat("ptr",matname),'w');
isok = (ptr >= 0);
if (!isok),
  error(sprintf('print_mat: fopen returns %s', msg));
  return;
end

[ind,msg] = fopen(strcat("ind",matname),'w');
isok = (ind >= 0);
if (!isok),
  error( sprintf('print_mat: fopen returns %s', msg));
  return;
end

[val,msg] = fopen( strcat("val",matname) , 'w');
isok = (val >= 0);
if (!isok),
  error( sprintf('print_mat: fopen returns %s', msg));
  return;
end

nrows = size(A,1);
ncols = size(A,2);
nnzA = nnz(A);
[ii,jj,aij] = find(A);


% -----------------------
% sort sparse matrix by rows and colums
% -----------------------

ipos = ii * ncols + jj;
[dummy, idx ] = sort( ipos);
clear dummy

ii = ii(idx);
jj = jj(idx);
aij = aij(idx);


% ----------------------
% print ptr
% ----------------------

cc = 0;
istat = fprintf(ptr,'%d ',cc);
isok = (istat >= 0);
if (!isok),
  error(sprintf('print_mat: fprintf returns istat=%d',istat));
  return;
end

for i=1:nrows-1,
  cc = cc + sum(ii == i);
  istat = fprintf(ptr,'%d ',cc);
  isok = (istat >= 0);
  if (!isok),
    error(sprintf('print_mat: fprintf returns istat=%d',istat));
    return;
  end;
end;

cc = cc + sum(ii == nrows);
if (cc != nnzA),
  sprintf('error calculating ptr');
end;
istat = fprintf(ptr,'%d',cc);
isok = (istat >= 0);
if (!isok),
  error(sprintf('print_mat: fprintf returns istat=%d',istat));
  return;
end;

istat = fclose(ptr);
isok = (istat == 0);
if (!isok),
  error(sprintf('print_mat: fclose returns istat=%d',istat));
  return;
end;


% ----------------------
% print ind and val
% ----------------------

for i=1:nnzA-1,
  istat = fprintf(ind,'%d ',jj(i)-1);
  isok = (istat >= 0);
  if (!isok),
    error(sprintf('print_mat: fprintf returns istat=%d',istat));
    return;
  end;

  istat = fprintf(val,'%1.17g ',aij(i));
  isok = (istat >= 0);
  if (!isok),
    error(sprintf('print_mat: fprintf returns istat=%d',istat));
    return;
  end;
end;

istat = fprintf(ind,'%d',jj(nnzA)-1);
isok = (istat >= 0);
if (!isok),
  error(sprintf('print_mat: fprintf returns istat=%d',istat));
  return;
end;

istat = fprintf(val,'%1.17g',aij(nnzA));
isok = (istat >= 0);
if (!isok),
  error(sprintf('print_mat: fprintf returns istat=%d',istat));
  return;
end;

istat = fclose(ind);
isok = (istat == 0);
if (!isok),
  error(sprintf('print_mat: fclose returns istat=%d',istat));
  return;
end;

istat = fclose(val);
isok = (istat == 0);
if (!isok),
  error(sprintf('print_mat: fclose returns istat=%d',istat));
  return;
end;

end
