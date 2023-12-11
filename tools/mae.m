function [MAE] = mae(Lhat,L,idx)
    L0_val = L(idx);
    Lhat_val = Lhat(idx);
    err =  Lhat_val(:) - L0_val(:);
    MAE = mean(abs(err));
end