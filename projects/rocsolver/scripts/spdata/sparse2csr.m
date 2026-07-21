function [Ap, Ai, Ax] = sparse2csr(A, is_one_based_in)
%
% [Ap, Ai, Ax] = sparse2csr(A, [is_one_based])
%
% convert matlab sparse matrix to
% Compressed sparse row format
%
is_one_based = 0;
if (nargin >= 2),
  is_one_based = is_one_based_in;
end;
is_zero_based = ~is_one_based;


nrow = size(A,1);
ncol = size(A,2);

nnzA = nnz(A);

Ap = zeros( 1,nrow+1);
Ai = zeros( 1,nnzA);
Ax = zeros( 1,nnzA);

% -----------------
% coordinate format
% -----------------
[ii,jj,aa] = find(A);

% ---------------------------
% sort to group rows together
% ---------------------------
[dummy, iperm] = sort( jj  + (ii-1) * ncol);
clear dummy;

ii = ii(iperm);
jj = jj(iperm);
aa = aa(iperm);

% -------------------------
% compute non-zeros per row
% -------------------------
nz_per_row = sum( A ~= 0, 2);

% -------------------------
% setup Ap(:) pointer array
% -------------------------
Ap(1) = 1;
Ap(2:(nrow+1)) = 1+cumsum( nz_per_row(1:nrow) );


% ------------------------------------------
% fill Ai(:) column indices and Ax(:) values
% ------------------------------------------
istart=1;
for irow=1:nrow,
 if (nz_per_row(irow) >= 1),
  iend = istart + nz_per_row(irow)-1;
  is_same_row = all( ii(istart:iend) == irow );
  if (~is_same_row),
    error(sprintf('sparse2csr: irow=%g,istart=%g,iend=%g', ...
                               irow,   istart,   iend ));
  end;
  Ai(istart:iend) = jj(istart:iend);
  Ax(istart:iend) = aa(istart:iend);

  istart = iend + 1;
 end;
end;


if (is_zero_based),
  Ap = Ap-1;
  Ai = Ai-1;
end;
