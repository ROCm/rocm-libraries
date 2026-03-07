function F = dft_matrix(n)
  
F = zeros(n,n);
omega_n = exp(-2*pi*j/n);
for i=1:n
  for j=1:n    
    F(i,j) = (omega_n^((i-1)*(j-1)));
  endfor
endfor



