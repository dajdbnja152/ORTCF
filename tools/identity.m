function [I] = identity(dim)
    I = zeros(dim);
    for i = 1:dim(1)
        I(i,i,1,1) = 1;
    end
end