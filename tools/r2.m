function [R2] = r2(Lhat,L,idx)
    L0_val = L(idx);
    Lhat_val = Lhat(idx);
    err = (Lhat_val(:) - L0_val(:)).^2;
    err0 = (mean(L0_val) - L0_val(:)).^2;
    R2 = 1-(sum(err)/sum(err0));
end