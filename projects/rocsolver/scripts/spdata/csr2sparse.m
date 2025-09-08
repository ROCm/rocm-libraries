function A = csr2sparse( Ap, Ai, Ax )
% A = csr2sparse( Ap, Ai, Ax )
%
% convert CSR matrix to matlab sparse matrix
%
n = max(size(Ap))-1;
nnz = Ap( n+1) - Ap(1);
ncol  = max( Ai(:)) + 1;
nrow = n;

ii = zeros(nnz,1);
ip = 1;
for irow=1:nrow,
  nz = Ap(irow+1)-Ap(irow);
  ii(ip:(ip+nz-1))  = irow;
  ip = ip + nz;
end;
A = sparse( ii,Ai+1,Ax, nrow,ncol );

