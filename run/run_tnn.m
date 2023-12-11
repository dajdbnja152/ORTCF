function [X_recov,RSE,RMSE,R2,time] = run_tnn(X_0,C_0,Omega_tra,Omega_val,Omega_tst,lambda_)
M_0 = C_0 + X_0 .* Omega_tra;
Omega_ = find(M_0~=0);
X_tst = X_0 .* Omega_tst;
X_val = X_0 .* Omega_val;
dim_ = size(X_0);
lambda_ = 1/sqrt(log(dim_(1)*dim_(1))).*lambda_;
for i = 1:length(lambda_)
    [X_recov,~,~,~,~] = lrtcR_tnn(M_0,Omega_,lambda_(i));
    X_recov_val = X_recov .* Omega_val;
    RMSE_val(i) = norm(X_recov_val(:)- X_val(:))/norm(X_val(:));
end
[~,idx_opt_] = min(RMSE_val);
if length(idx_opt_) >= 2
    idx_opt = idx_opt_(1);
else
    idx_opt = idx_opt_;
end
lambda_opt = lambda_(idx_opt);
tic
[X_recov,~,~,~,~] = lrtcR_tnn(M_0,Omega_,lambda_opt);
time = toc;
X_recov_tst = X_recov.* Omega_tst;
RSE = norm(X_recov_tst(:)- X_tst(:))/norm(X_tst(:));
idx_tst = find(Omega_tst);
RMSE = rmse(X_recov_tst,X_tst,idx_tst);
R2 = r2(X_recov_tst,X_tst,idx_tst);
end