function [X_recov,RSE,RMSE,R2,time] = run_olstec(X_0,C_0,Omega_tra,Omega_val,Omega_tst)
rank = 1;
M_0 = C_0 + X_0 .* Omega_tra;
M_0 = double(permute(M_0,[1,3,2]));
X_0 = double(permute(X_0,[1,3,2]));
Omega_tra = double(permute(Omega_tra,[1,3,2]));
Omega_val = double(permute(Omega_val,[1,3,2]));
Omega_tst = double(permute(Omega_tst,[1,3,2]));
tensor_dims = size(M_0);
options.verbose = 1;
lambda_ = [0.1,0.5,1,5];
for j = 1:length(lambda_)
    options.lambda = lambda_(j);
    [Xsol_olstec, infos_olstec, sub_infos_olstec,~] = olstec(M_0, Omega_tra, [], tensor_dims, rank, [], options);
    L = reshape(sub_infos_olstec.L,size(M_0));
    L_val = L .* Omega_val;
    X_0_val = X_0 .* Omega_val;
    RMSE_L_val(j) = norm(L_val(:)- X_0_val(:))/norm(X_0_val(:));
  
end
[~,idx_opt_] = min(RMSE_L_val);
if length(idx_opt_) >= 2
    idx_opt = idx_opt_(1);
else
    idx_opt = idx_opt_;
end
options.lambda = lambda_(idx_opt);
[Xsol_olstec, infos_olstec, sub_infos_olstec,time] = olstec(M_0, Omega_tra, [], tensor_dims, rank, [], options);
X_recov = reshape(sub_infos_olstec.L,size(M_0));
X_recov_tst = X_recov .* Omega_tst;
X_tst = X_0 .* Omega_tst;
idx_tst = find(Omega_tst);
RSE = norm(X_recov_tst(:)- X_tst(:))/norm(X_tst(:));
RMSE = rmse(X_recov_tst,X_tst,idx_tst);
R2 = r2(X_recov_tst,X_tst,idx_tst);
end
