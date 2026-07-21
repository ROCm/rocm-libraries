% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************

function isok = print_gvec( filename, A )
%
%  isok = print_gvec(filename, A)
%
%  This function prints into file the contents of full vector A.
%  Inputs:
%  filename    The name of the generated file
%  A           The full vector (column or row) to be printed
%
%  Outputs:
%  isok        This will be 0 if any of the file operations failed
%
%  The resulting file is created in the working directory.
%
%  (This script is for internal use only. It is not part of
%  rocSOLVER library interface and could change or be removed
%  without any notice)

[fid,msg] = fopen(filename,'w');
isok_fid = (fid >= 0);
if (~isok_fid),
  error(sprintf('print_vec:fopen %s returns %s', ...
                           filename,   msg ));
  return;
end;

istat_rewind = frewind(fid);

nitems = prod(size(A));

is_integer = all( floor(A(:)) == A(:) );
if (is_integer),
  numbytes = fprintf(fid, ' %d', reshape(A(:),1,nitems) );
else
  numbytes = fprintf(fid, ' %25.20e', reshape(A(:),1,nitems) );
end;

numbytes = numbytes + fprintf(fid,'\n');
istat_close = fclose( fid );

isok = (istat_rewind >= 0) & (numbytes >= 0) & (istat_close >= 0);





end
