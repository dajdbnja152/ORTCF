function [X_recov,RSE,RMSE,R2,time] = run_otrpca(X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda_,transform)
loop = 1;
lambda_1 = 0.01;
dim_ = size(X_0);
lambda_ = 1/sqrt(log(dim_(1)*dim_(1)))*lambda_;
M_0 = C_0 + X_0 .* Omega_tra;
X_tst = X_0 .* Omega_tst;
X_val = X_0 .* Omega_val;
for i = 1:length(lambda_)
    [X_recov,~,~] = otrpca(M_0,rank,loop,lambda_1,lambda_(i),transform);
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
[X_recov,~,time] = otrpca(M_0,rank,loop,lambda_1,lambda_opt,transform);
X_recov_tst = X_recov .* Omega_tst;
RSE = norm(X_recov_tst(:)- X_tst(:))/norm(X_tst(:));
idx_tst = find(Omega_tst);
RMSE = rmse(X_recov_tst,X_tst,idx_tst);
R2 = r2(X_recov_tst,X_tst,idx_tst);
end