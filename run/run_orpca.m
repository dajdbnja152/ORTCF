function [X_recov,RSE,RMSE,R2,time] = run_orpca(X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda_2_)
n = 0;
dim_ = size(X_0);
lambda_1 = 0.01;
lambda_2 = 1/sqrt(log(dim_(1)*dim_(1))).*lambda_2_;
M_0 = C_0 + X_0 .* Omega_tra;
X_0_mat = ten2mat(X_0);
Omega_tst_mat = ten2mat(Omega_tst);
Omega_val_mat = ten2mat(Omega_val);
M_0_mat = ten2mat(M_0);
RMSE_val = ones(length(lambda_2),1);
X_mat_val = X_0_mat .* Omega_val_mat;
X_mat_tst = X_0_mat .* Omega_tst_mat;
for i = 1: length(lambda_2)
        n = n + 1;
        [X_recov_val,~,~] = orpca(M_0_mat,rank,lambda_1,lambda_2(i));
        X_recov1 = X_recov_val .* Omega_val_mat;
        RMSE_val(n) = norm(X_recov1(:)- X_mat_val(:))/norm(X_mat_val(:));
end
[~,idx_opt_] = min(RMSE_val);
if length(idx_opt_) >= 2
    idx_opt = idx_opt_(1);
else
    idx_opt = idx_opt_;
end
lambda_2_opt = lambda_2(idx_opt);
[X_recov,~,time] = orpca(M_0_mat,rank,lambda_1,lambda_2_opt);
X_recov_tst = X_recov .* Omega_tst_mat;
RSE = norm(X_recov_tst(:)- X_mat_tst(:))/norm( X_mat_tst(:));
idx_tst = find(Omega_tst_mat);
RMSE = rmse(X_recov_tst, X_mat_tst,idx_tst);
R2 = r2(X_recov_tst, X_mat_tst,idx_tst);
end