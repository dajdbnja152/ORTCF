function [X_recov,RSE,RMSE,R2,time] = run_ormcf(T,X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda_1_,lambda_3_)
n = 0;
loop = 1;
dim_ = size(X_0);
lambda_1_ = 1/sqrt(log(dim_(1)*dim_(1))).*lambda_1_;
M_0 = C_0 + X_0 .* Omega_tra;
ten_x = T(:,1:31,:);
Mat = ten2mat(ten_x);
[U_m,~,~] = svd(Mat);
U1_m = X_side_info(U_m,10,0);
X_0_mat = ten2mat(X_0);
Omega_tra_mat = ten2mat(Omega_tra);
Omega_tst_mat = ten2mat(Omega_tst);
Omega_val_mat = ten2mat(Omega_val);
M_0_mat = ten2mat(M_0);
lambda_1 = ones(length(lambda_1_)*length(lambda_3_));
lambda_3 = ones(length(lambda_1_)*length(lambda_3_));
RMSE_val = ones(length(lambda_1_)*length(lambda_3_));
X_mat_val = X_0_mat .* Omega_val_mat;
X_mat_tst = X_0_mat .* Omega_tst_mat;
for i = 1: length(lambda_1_)
    for j = 1: length(lambda_3_)
        n = n + 1;
        [X_recov_val,~,~,~] = ormcf(M_0_mat,U1_m,Omega_tra_mat,rank,loop,0.01,lambda_1_(i),lambda_3_(j));
        X_recov1 = X_recov_val .* Omega_val_mat;
        RMSE_val(n) = norm(X_recov1(:)- X_mat_val(:))/norm(X_mat_val(:));
        lambda_1(n) = lambda_1_(i);
        lambda_3(n) = lambda_3_(j);
    end
end
[~,idx_opt_] = min(RMSE_val);
if length(idx_opt_) >= 2
    idx_opt = idx_opt_(1);
else
    idx_opt = idx_opt_;
end
lambda_1_opt = lambda_1(idx_opt);
lambda_3_opt = lambda_3(idx_opt);
[X_recov,~,~,time] = ormcf(M_0_mat,U1_m,Omega_tra_mat,rank,loop,0.01,lambda_1_opt,lambda_3_opt);
X_recov_tst = X_recov .* Omega_tst_mat;
RSE = norm(X_recov_tst(:)- X_mat_tst(:))/norm( X_mat_tst(:));
idx_tst = find(Omega_tst_mat);
RMSE = rmse(X_recov_tst, X_mat_tst,idx_tst);
R2 = r2(X_recov_tst, X_mat_tst,idx_tst);
end