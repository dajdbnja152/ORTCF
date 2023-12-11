function [X,S,time] = orpca(Z,d,lambda1,lambda2)
[p,n] = size(Z);
L = randn(p, d);
A = zeros(d, d);
B = zeros(p, d);
R = zeros(n, d);
S = zeros(p,n);
for t=1:n
  tic;
  z = Z(:, t);
  [r, e] = solve_proj2(z, L, lambda1, lambda2);
  A = A + r * r';
  B = B + (z-e) * r';
  L = update_col_orpca(L, A, B, lambda1);
  R(t, :) = r';
  S(:, t) = e;
  time = toc;
end
X = L * R';
end