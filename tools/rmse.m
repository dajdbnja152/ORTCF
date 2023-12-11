function [RMSE] = rmse(Lhat,L,idx)
    L0_val = L(idx);
    Lhat_val = Lhat(idx);
    err = Lhat_val(:) - L0_val(:);
    RMSE = sqrt(mean(err.^2));
end