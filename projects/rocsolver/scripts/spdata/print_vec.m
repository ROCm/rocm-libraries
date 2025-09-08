% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function isok = print_vec( vecname, v )

%{
-----------------------------------------------------------
  This function prints into file the contents of full vector v.

  isok = print_vec(vecname, v)

  Inputs:
  vecname     The name of the generated file
  v           The full vector (column or row) to be printed

  Outputs:
  isok        This will be -1 if any of the file operations failed

  The resulting file is created in the working directory.

  (This script is for internal use only. It is not part of
  rocSOLVER library interface and could change or be removed
  without any notice)
-----------------------------------------------------------
%}



[vec,msg] = fopen( vecname , 'w');
isok = (vec >= 0);
if (!isok),
  error( sprintf('print_vec: fopen returns %s', msg));
  return;
end;

n = max(size(v,1),size(v,2));
for i=1:n-1,
  istat = fprintf(vec,'%d ',v(i)-1);
  isok = (istat >= 0);
  if (!isok),
    error(sprintf('print_vec: fprintf returns istat=%d',istat));
    return;
  end;
end;

istat = fprintf(vec,'%d',v(n)-1);
isok = (istat >= 0);
if (!isok),
  error(sprintf('print_vec: fprintf returns istat=%d',istat));
  return;
end;

istat = fclose(vec);
isok = (istat == 0);
if (!isok),
  error(sprintf('print_vec: fclose returns istat=%d',istat));
  return;
end;

end

