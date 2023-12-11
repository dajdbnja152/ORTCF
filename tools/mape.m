function [MAPE] = mape(Lhat,L,idx)
    L0_val = L(idx);
    Lhat_val = Lhat(idx);
    err = Lhat_val(:) - L0_val(:);
    MAPE = mean(abs(err./L0_val)).*100;
end