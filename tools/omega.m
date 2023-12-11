function [Omega]=omega(M,p)
Nways = size(M);
m = round(p*prod(Nways));
temp = randperm(prod(Nways));
Omega = zeros(Nways);
Omega(temp(1:m)) = 1;
end