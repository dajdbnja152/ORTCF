function [X_recov,RSE,RMSE,R2,time] = run_olrtr(X_0,C_0,Omega_tra,Omega_val,Omega_tst,lambda_)
[n1,T,n2] = size(X_0);
lambda1 = 0.01;
lambda2 = 1/sqrt(log(n1*n1));
nrank = 3;
outlier_dim = 2;
M_0 = C_0 + X_0 .* Omega_tra;
Rec = [];
lambda_ = lambda_.* lambda2;
for j = 1: length(lambda_)
X_0_t = double(permute(X_0,[1,3,2]));
M_0_t = double(permute(M_0,[1,3,2]));
Omega_ = 1 - Omega_tra;
Omega_t_ = double(permute(Omega_,[1,3,2]));
Omega_tst_t = double(permute(Omega_tst,[1,3,2]));
Omega_val_t = double(permute(Omega_val,[1,3,2]));
Rec = [];
X_recov = zeros([n1,n2,T]);
for i = 1:T
    D = M_0_t(:, :, i);
    Sigma_bar = Omega_t_(:, :,i);
    D = squeeze(D);  
    [Xhat,~,~, Rec] = OLRTR(D, lambda1, lambda_(j), Rec, Sigma_bar, nrank, outlier_dim);
    X_recov(:,:,i) = Xhat;
end 
Xhat_OL_val = X_recov .* Omega_val_t;
X_0_val = X_0_t .* Omega_val_t;
RMSE_L_val(j) = norm(Xhat_OL_val(:)- X_0_val(:))/norm(X_0_val(:));
end
[~,idx_opt_] = min(RMSE_L_val);
if length(idx_opt_) >= 2
    idx_opt = idx_opt_(1);
else
    idx_opt = idx_opt_;
end
lambda_3_opt = lambda_(idx_opt);
X_recov = zeros([n1,n2,T]);
for i = 1:T
    tic
    D = M_0_t(:, :, i);
    Sigma_bar = Omega_t_(:, :,i);
    D = squeeze(D);  
    Rec_old = Rec;
    [Xhat, ~, ~, Rec] = OLRTR(D, lambda1, lambda_3_opt, Rec, Sigma_bar, nrank, outlier_dim);
    X_recov(:,:,i) = Xhat;
    time = toc;
end 

X_tst = X_0_t .* Omega_tst_t;
X_recov_tst = X_recov .* Omega_tst_t;
RSE = norm(X_recov_tst(:)- X_tst(:))/norm(X_tst(:));
idx_tst = find(Omega_tst_t);
RMSE = rmse(X_recov_tst,X_tst,idx_tst);
R2 = r2(X_recov_tst,X_tst,idx_tst);
end