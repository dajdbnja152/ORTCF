function [X_recov,RSE,RMSE,R2,time] = run_OSTD(X_0,C_0,Omega_tra,Omega_val,Omega_tst,lambda_)
[n1,T,n2] = size(X_0);
k = 1;
lambda1 = 0.01;
lambda_ = lambda_.* 1/sqrt(log(n1*n1));
X_recov = zeros(size(X_0));
S = zeros(size(X_0));
M_0 = C_0 + X_0 .* Omega_tra;
for j = 1: length(lambda_)
for i = 1:T
    frame = M_0(:,i,:);
    frame = tensor(frame);
    if(k == 1) 
      Tm = []; 
    end
    [Tlowrank,Tsparse,Tmask,Tm] = OSTD(frame,k,Tm,lambda1,lambda_(j));
    X_recov(:,i,:) = Tlowrank;
    S(:,i,:) = Tsparse;
    k = k+1;    
end
L_val = X_recov .* Omega_val;
X_0_val = X_0 .* Omega_val;
S = S .* Omega_tra;
RMSE_L_val(j) = norm(L_val(:)- X_0_val(:))/norm(X_0_val(:));
end
[~,idx_opt_] = min(RMSE_L_val);
if length(idx_opt_) >= 2
    idx_opt = idx_opt_(1);
else
    idx_opt = idx_opt_;
end
lambda2_opt = lambda_(idx_opt);
X_recov = zeros(size(X_0));
for i = 1:T
    frame = M_0(:,i,:);
    frame = tensor(frame);
    if(k == 1) 
      Tm = []; 
    end
    tic
    [Tlowrank,Tsparse,Tmask,Tm] = OSTD(frame,k,Tm,lambda1,lambda2_opt);
    time = toc;
    X_recov(:,i,:) = Tlowrank;
    S(:,i,:) = Tsparse;
    k = k+1;    
end
X_recov_tst = X_recov .* Omega_tst;
X_tst = X_0 .* Omega_tst;
idx_tst = find(Omega_tst);
RSE = norm(X_recov_tst(:)- X_tst(:))/norm(X_tst(:));
RMSE = rmse(X_recov_tst,X_tst,idx_tst);
R2 = r2(X_recov_tst,X_tst,idx_tst);
end
