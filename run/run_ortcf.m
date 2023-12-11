function [X_recov,RSE,RMSE,R2,time,diff] = run_ortcf(T,X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda_1_,lambda_3_,transform)
n = 0;
loop = 1;
dim_ = size(X_0);
lambda_1_ = 1/sqrt(log(dim_(1)*dim_(1))).*lambda_1_;
M_0 = C_0 + X_0 .* Omega_tra;
ten_x = T(:,1:31,:);
[U_tmp,~,~] = tsvd(ten_x,transform);
U = U_tmp(:,1:10,:);
X_tst = X_0 .* Omega_tst;
X_val = X_0 .* Omega_val;
RMSE_val = ones(length(lambda_3_)*length(lambda_3_));
lambda_1 = zeros(length(lambda_1_)*length(lambda_3_));
lambda_3 = zeros(length(lambda_1_)*length(lambda_3_));
for i = 1: length(lambda_1_)
    for j = 1: length(lambda_3_)
        n = n + 1;
        [X_recov,~,~,~] = ortcf(M_0,U,Omega_tra,rank,loop,0.01,lambda_1_(i),lambda_3_(j),transform);
        X_recov_val = X_recov .* Omega_val;
        RMSE_val(n) = norm(X_recov_val(:)- X_val(:))/norm(X_val(:));
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
[X_recov,S,~,time,diff] = ortcf(M_0,U,Omega_tra,rank,loop,0.01,lambda_1_opt,lambda_3_opt,transform);
X_recov_tst = X_recov .* Omega_tst;
RSE = norm(X_recov_tst(:)- X_tst(:))/norm(X_tst(:));
idx_tst = find(Omega_tst);
RMSE = rmse(X_recov_tst,X_tst,idx_tst);
R2 = r2(X_recov_tst,X_tst,idx_tst);
end