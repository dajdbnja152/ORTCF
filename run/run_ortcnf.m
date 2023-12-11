function [X_recov,RMSE_L1_,RMSE_S1_] = run_ortcnf(X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda_)
n = 0;
loop = 1;
lambda_2 = 15;
lambda_5 = 1; %0.01
M_0 = C_0 + X_0 .* Omega_tra;
% df_x = readtable("PeMS_6_online.xlsx");
% df_x = table2array(df_x);
% ten_x = zeros(228,30,10);
% for i = 1:length(df_x)
%     ten_x(df_x(i,4),df_x(i,1),df_x(i,3)) = df_x(i,2);
% end
load('PeMS08_10.mat')
ten_x = T(:,1:31,:);
dim_ = size(ten_x);
transformH.L = 'fft';transformH.rho = prod(dim_(3:end));
[U_tmp,~,~] = H_tsvd(ten_x,transformH);
U = U_tmp(:,1:10,:);
X_tst = X_0 .* Omega_tst;
X_val = X_0 .* Omega_val;
RMSE_L_tst = ones(length(lambda_)*length(lambda_));
RMSE_S_tst = ones(length(lambda_)*length(lambda_));
RMSE_val = ones(length(lambda_)*length(lambda_));
lambda_m_ = zeros(length(lambda_)*length(lambda_));
lambda_n_ = zeros(length(lambda_)*length(lambda_));
for i = 1: length(lambda_)
    for j = 1: length(lambda_)
        n = n + 1;
        [X_recov,S1,~] = ortcnf(M_0,U,Omega_tra,rank,loop,lambda_2,lambda_(i),lambda_(j),lambda_5);
        X_recov_val = X_recov .* Omega_val;
        X_recov_tst = X_recov .* Omega_tst;
        S1 = S1 .* Omega_tra;
        RMSE_val(n) = norm(X_recov_val(:)- X_val(:))/norm(X_val(:));
        RMSE_L_tst(n) = norm(X_recov_tst(:)- X_tst(:))/norm(X_tst(:));
        RMSE_S_tst(n) = norm(S1(:)- C_0(:))/norm(C_0(:));
        lambda_m_(n) = lambda_(i);
        lambda_n_(n) = lambda_(j);
        fprintf("lambda_2:%.5f,lambda_3:%.5f,lambda_5:%.5f,RMSE_L1:%.5f,RMSE_tst:%.5f\n",lambda_(i),lambda_(j),lambda_5,RMSE_val(n),RMSE_L_tst(n));
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
[X_recov,~,~] = ortcnf(M_0,U,Omega_tra,rank,loop,lambda_2,lambda_m_opt,lambda_n_opt,lambda_5);
RMSE_L1_ = RMSE_L_tst(idx_opt);
RMSE_S1_ = RMSE_S_tst(idx_opt);
fprintf("Ours RSE_L:%.5f\n",RMSE_L1_);
fprintf("Ours RSE_S:%.5f\n",RMSE_S1_);
%fprintf("lambda_2:%.5f,lambda_3:%.5f,lambda_5:%.5f\n",lambda_m_opt,lambda_n_opt,lambda_5);
end