function out = convert_3d_to_1d(in, ordering)

if isvector(in)
  error('Invalid input');
endif

if strcmp(ordering,'column-major')
  out = reshape(in, 1, []);  
  out=conj(out);
elseif strcmp(ordering,'row-major')
  out = reshape(in', 1, []);
else
  error('Invalid option');
endif
