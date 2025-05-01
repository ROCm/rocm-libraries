% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function isok = print_x( matname, X )

%{
-----------------------------------------------------------
  This function prints into file the contents of full matrix M.

  isok = print_vec(matname, X)

  Inputs:
  matname     The name of the generated file
  X           The full matrix to be printed

  Outputs:
  isok        This will be -1 if any of the file operations failed

  The resulting file is created in the working directory.

  (This script is for internal use only. It is not part of
  rocSOLVER library interface and could change or be removed
  without any notice)
-----------------------------------------------------------
%}


[vec,msg] = fopen( matname , 'w');
isok = (vec >= 0);
if (!isok),
  error( sprintf('print_x: fopen returns %s', msg));
  return;
end;

m = size(X,1);
n = size(X,2);
for i=1:m,
  for j=1:n-1,
    istat = fprintf(vec,'%1.17g ',X(i,j));
    isok = (istat >= 0);
    if (!isok),
      error(sprintf('print_x: fprintf returns istat=%d',istat));
      return;
    end;
  end;

  istat = fprintf(vec,'%1.17g',X(i,n));
  isok = (istat >= 0);
  if (!isok),
    error(sprintf('print_x: fprintf returns istat=%d',istat));
    return;
  end;

  istat = fprintf(vec,'\n');
  isok = (istat >= 0);
  if (!isok),
    error(sprintf('print_x: fprintf returns istat=%d',istat));
    return;
  end;
end;

istat = fclose(vec);
isok = (istat == 0);
if (!isok),
  error(sprintf('print_x: fclose returns istat=%d',istat));
  return;
end;


end

