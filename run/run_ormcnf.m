function [X_recov,RMSE_L6_,RMSE_S6_] = run_ormcnf(X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda_)
n = 0;
loop = 1;
M_0 = C_0 + X_0 .* Omega_tra;
% df_x = readtable("PeMS_6_online.xlsx");
% df_x = table2array(df_x);
% ten_x = zeros(228,30,10);
% for i = 1:length(df_x)
%     ten_x(df_x(i,4),df_x(i,1),df_x(i,3)) = df_x(i,2);
% end
load('PeMS08_10.mat')
ten_x = T(:,1:31,:);
Mat = ten2mat(ten_x);
[U_m,~,~] = svd(Mat);
U1_m = X_side_info(U_m,10,0);
lambda_2 = 10;
lambda_5 = 0.01;
X_0_mat = ten2mat(X_0);
Omega_tra_mat = ten2mat(Omega_tra);
Omega_tst_mat = ten2mat(Omega_tst);
Omega_val_mat = ten2mat(Omega_val);
M_0_mat = ten2mat(M_0);
C_0_mat = ten2mat(C_0);
RMSE_val = ones(length(lambda_)*length(lambda_));
lambda_m_ = zeros(length(lambda_)*length(lambda_));
lambda_n_ = zeros(length(lambda_)*length(lambda_));
RMSE_L_tst = ones(length(lambda_)*length(lambda_));
RMSE_S_tst = ones(length(lambda_)*length(lambda_));
X_mat_val = X_0_mat .* Omega_val_mat;
X_mat_tst = X_0_mat .* Omega_tst_mat;

for i = 1: length(lambda_)
    for j = 1: length(lambda_)
        n = n + 1;
        [X_recov_val,S6,~] = ormcnf(M_0_mat,U1_m,Omega_tra_mat,rank,loop,lambda_2,lambda_(i),lambda_(j),lambda_5);
        X_recov1 = X_recov_val .* Omega_val_mat;
        X_recov2 = X_recov_val .* Omega_tst_mat;
        S6 = S6 .* Omega_tra_mat;
        RMSE_val(n) = norm(X_recov1(:)- X_mat_val(:))/norm(X_mat_val(:));
        RMSE_L_tst(n) = norm(X_recov2(:)- X_mat_tst(:))/norm(X_mat_tst(:));
        RMSE_S_tst(n) = norm(S6(:) - C_0_mat(:))/norm(C_0_mat(:));
        lambda_m_(n) = lambda_(i);
        lambda_n_(n) = lambda_(j);
        %fprintf("lambda_2:%.5f,lambda_3:%.5f,lambda_5:%.5f,RMSE_L1:%.5f,RMSE_tst:%.5f\n",lambda_(i),lambda_(j),lambda_5,RMSE_val(n),RMSE_L_tst(n));
    end
end
[~,idx_opt_] = min(RMSE_val);
if length(idx_opt_) >= 2
    idx_opt = idx_opt_(1);
else
    idx_opt = idx_opt_;
end
lambda_m_opt = lambda_m_(idx_opt);
lambda_n_opt = lambda_n_(idx_opt);
[X_recov,~,~] = ormcnf(M_0_mat,U1_m,Omega_tra_mat,rank,loop,lambda_2,lambda_m_opt,lambda_n_opt,lambda_5);
RMSE_L6_ = RMSE_L_tst(idx_opt);
RMSE_S6_ = RMSE_S_tst(idx_opt);
fprintf("Matrix RSE_L:%.5f\n",RMSE_L6_);
fprintf("Matrix RSE_S:%.5f\n",RMSE_S6_);
%fprintf("lambda_2:%.5f,lambda_3:%.5f,lambda_5:%.5f\n",lambda_m_opt,lambda_n_opt,lambda_5);
end