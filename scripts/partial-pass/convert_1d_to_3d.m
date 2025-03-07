function out = convert_1d_to_3d(in, n1, n2, n3, batch, ordering)

if ~isvector(in)
  error('Invalid input');
endif

if strcmp(ordering,'column-major')
  out=reshape(in, n3, n2, n1, batch);
elseif strcmp(ordering,'row-major')
  out=reshape(in, n1, n2, n3, batch)';
else
  error('Invalid option');
endif




