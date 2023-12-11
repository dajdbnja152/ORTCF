function [S]=S_noisy(M,p)
[n1,n2,n3,n4] =size(M);
m = p*n1*n2*n4;
temp = rand(n1*n2*n3*n4,1);
[B,I] = sort(temp);
I = I(1:m);
Omega = zeros(n1,n2,n3,n4);
Omega(I) = 1;
E = sign(rand(n1,n2,n3,n4)-0.5);
S = Omega.*E; % sparse part, or noises. S = P_Omega(E)
end