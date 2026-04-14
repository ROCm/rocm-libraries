% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************
function A = read_vec( filename )
%
% A = read_vec(  filename  )
%
% read a vector from file

[fid,msg] = fopen(filename,'r');
isok_fid = (fid >= 0);
if (~isok_fid),
  error(sprintf('read_vec:fopen %s returns %s', ...
                           filename,   msg ));
  return;
end;

istat_rewind = frewind(fid);
isok = (istat_rewind == 0);
if (~isok),
  error(sprintf('read_vec: error in frewind for filename=%s', ...
                                                filename ));
  return;
end;

use_fscanf = 1;
if (use_fscanf),
  [A,count,errmsg] = fscanf( fid, ' %g' );
else

  line = fgets( fid );
  is_float = index(line,'.') | index(line,'e') | index(line,'E');
  if (is_float),
  else
    [A,count,errmsg] = sscanf( line, ' %d' );
  end;

end;

idebug = 0;
if (idebug >= 1),
 if (numel(errmsg) >= 1),
  error(sprintf('read_vec: is_float=%d,filename=%s, count=%d, errmsg=%s', ...
                           is_float,   filename,    count,    errmsg ));
  return;
 end;
end;

status = fclose(fid);
isok = (status == 0);
if (~isok),
  error(sprintf('read_vec: error with fclose for file %s', ...
                                                 filename ));
  return;
end;



end
