function [X_recov,RSE,RMSE,R2,time] = run_grasta(X_0,C_0,Omega_tra,Omega_val,Omega_tst)
[n1,T,n2] = size(X_0);
M_0 = C_0 + X_0 .* Omega_tra;
M_0 = double(permute(M_0,[1,3,2]));
X_0 = double(permute(X_0,[1,3,2]));
C_0 = double(permute(C_0,[1,3,2]));
Omega_tra = double(permute(Omega_tra,[1,3,2]));
Omega_tst = double(permute(Omega_tst,[1,3,2]));
Omega_val = double(permute(Omega_val,[1,3,2]));
rank = 10;
num_params_of_tensor = rank * sum(size(M_0),2);
numr = n1*n2;
numc = T;
matrix_rank = floor(num_params_of_tensor/(numr+numc));
if matrix_rank < 1
        matrix_rank = 1;
end
M_0 = reshape(M_0,[n1*n2,T]);
X_0 = reshape(X_0,[n1*n2,T]);
C_0 = reshape(C_0,[n1*n2,T]);
Omega_tra = reshape(Omega_tra,[n1*n2,T]);
Omega_tst = reshape(Omega_tst,[n1*n2,T]);
Omega_val = reshape(Omega_val,[n1*n2,T]);
options.maxepochs           = 1;
options.tolcost             = 1e-8;
options.permute_on          = false;    
options.verbose             = 1;
options.store_subinfo       = true;     
options.RANK                = matrix_rank;
options.rho                 = 1.8;    
options.MAX_MU              = 10000; % set max_mu large enough for initial subspace training
options.MIN_MU              = 1;
options.ITER_MAX            = 20; 
options.DIM_M               = n1*n2;  % your data's dimension
options.USE_MEX             = 0; 
options.store_matrix        = true;
lambda_ = [0.1,0.5,1,5];
for j = 1:length(lambda_)
    options.lambda = lambda_(j);
    [Xsol_grasta, infos_grasta, sub_infos_grasta, ~,~] = grasta([], M_0, Omega_tra, [], n1*n2, T, options);
    L = sub_infos_grasta.L;
    L_val = L .* Omega_val;
    X_0_val = X_0 .* Omega_val;
    RMSE_L_val(j) = norm(L_val(:)- X_0_val(:))/norm(X_0_val(:));
    
[~,idx_opt_] = min(RMSE_L_val);
if length(idx_opt_) >= 2
    idx_opt = idx_opt_(1);
else
    idx_opt = idx_opt_;
end

options.lambda = lambda_(idx_opt);
[Xsol_grasta, infos_grasta, sub_infos_grasta, ~,time] = grasta([], M_0, Omega_tra, [], n1*n2, T, options);
X_recov = reshape(sub_infos_grasta.L,size(M_0));
X_recov_tst = X_recov .* Omega_tst;
X_tst = X_0 .* Omega_tst;
idx_tst = find(Omega_tst);
RSE = norm(X_recov_tst(:)- X_tst(:))/norm(X_tst(:));
RMSE = rmse(X_recov_tst,X_tst,idx_tst);
R2 = r2(X_recov_tst,X_tst,idx_tst);
end
